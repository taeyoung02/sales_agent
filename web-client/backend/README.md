# Interactive AI Dealer Backend

FastAPI backend for the Interactive AI Dealer 3D vehicle simulator.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

3. Run the server:
```bash
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /api/health` - Health check
- `POST /api/chat` - Chat endpoint (to be implemented)
- `POST /api/scene/camera` - Camera control endpoint (to be implemented)

## Project Structure

```
backend/
├── main.py                 # FastAPI app entry point
├── rag.py                  # RAG pipeline
├── llm_client.py          # LLM client
├── camera_controller.py    # Camera control logic
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables (not in git)
```

