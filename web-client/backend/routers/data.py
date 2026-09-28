"""
Data Loader API router
Qdrant 벡터 데이터베이스에 차량 데이터를 로드하는 API
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import sys
import traceback
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas.schemas import LoadDataRequest, LoadDataResponse, DataStatusResponse
from rag.data_loader import load_vehicle_data_to_qdrant, load_vector_files
from rag.rag import get_rag_pipeline
from utils.path_utils import get_source_root

router = APIRouter(prefix="/api/data", tags=["data"])


@router.post("/load", response_model=LoadDataResponse)
async def load_data(request: LoadDataRequest):
    """
    Load vehicle vector data from JSON files into Qdrant

    Args:
        request: Load data request with optional data_dir and clear_existing flag

    Returns:
        Load data response with number of points loaded and total points
    """
    try:
        # Determine data directory
        if request.data_dir:
            data_dir = Path(request.data_dir)
        else:
            data_dir = get_source_root() / "car_data"

        if not data_dir.exists():
            raise HTTPException(
                status_code=404, detail=f"Data directory not found: {data_dir}"
            )

        # Load data to Qdrant asynchronously to prevent blocking the event loop
        # This allows the server to handle other requests while loading data
        # Use asyncio.to_thread to run the synchronous function in a thread pool
        points_loaded = await asyncio.to_thread(
            load_vehicle_data_to_qdrant,
            data_dir=data_dir,
            clear_existing=request.clear_existing,
        )

        # Get total points count
        rag = get_rag_pipeline()
        total = rag.client.count(collection_name=rag.collection_name, exact=True)

        return LoadDataResponse(
            success=True,
            message=f"Successfully loaded {points_loaded} points to Qdrant",
            points_loaded=points_loaded,
            total_points=total.count,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to load data: {str(e)}")


@router.get("/status", response_model=DataStatusResponse)
async def get_data_status():
    """
    Get current status of the Qdrant collection

    Returns:
        Data status response with collection name, total points, and status
    """
    try:
        rag = get_rag_pipeline()
        total = rag.client.count(collection_name=rag.collection_name, exact=True)

        return DataStatusResponse(
            collection_name=rag.collection_name,
            total_points=total.count,
            status="available" if total.count > 0 else "empty",
        )
    except Exception as e:
        print(f"Error getting data status: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Failed to get data status: {str(e)}"
        )
