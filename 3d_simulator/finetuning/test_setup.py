"""
Test script to verify dataset and model setup
"""
import torch
import numpy as np
from dataset import VehicleSegDataset
from model import SAMCLIPModel
import os


def test_dataset():
    """데이터셋 로딩 테스트"""
    print("="*50)
    print("Testing Dataset...")
    print("="*50)
    
    try:
        dataset = VehicleSegDataset(
            root_dir='./VehicleSeg10K',
            split='train'
        )
        
        print(f"✓ Dataset loaded successfully")
        print(f"  - Total samples: {len(dataset)}")
        print(f"  - Number of classes: {len(dataset.LABELS)}")
        print(f"  - Classes: {', '.join(dataset.LABELS)}")
        
        # 첫 번째 샘플 로드
        sample = dataset[0]
        print(f"\n✓ First sample loaded successfully")
        print(f"  - Image shape: {np.array(sample['image']).shape}")
        print(f"  - Number of masks: {len(sample['masks'])}")
        print(f"  - Labels: {sample['labels']}")
        
        return True
    
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False


def test_model():
    """모델 로딩 및 forward pass 테스트"""
    print("\n" + "="*50)
    print("Testing Model...")
    print("="*50)
    
    try:
        # CUDA 체크
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  - Using device: {device}")
        
        if device == 'cuda':
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA version: {torch.version.cuda}")
        
        # SAM checkpoint 체크
        sam_checkpoint = './checkpoints/sam_vit_b_01ec64.pth'
        if not os.path.exists(sam_checkpoint):
            print(f"✗ SAM checkpoint not found at: {sam_checkpoint}")
            return False
        
        print(f"✓ SAM checkpoint found")
        
        # 모델 로드
        print(f"  - Loading model...")
        model = SAMCLIPModel(
            sam_checkpoint=sam_checkpoint,
            sam_model_type='vit_b',
            clip_model_name='ViT-B-32',
            clip_pretrained='openai',
            num_classes=14,
            freeze_sam=True,
            device=device
        )
        
        print(f"✓ Model loaded successfully")
        
        # 더미 데이터로 forward pass 테스트
        print(f"  - Testing forward pass...")
        dummy_image = torch.randn(3, 480, 640).to(device)
        dummy_mask = torch.randint(0, 2, (480, 640)).float().to(device)
        
        with torch.no_grad():
            output = model(dummy_image, [dummy_mask])
        
        print(f"✓ Forward pass successful")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Expected shape: (1, 14)")
        
        # 메모리 정보
        if device == 'cuda':
            print(f"\n  GPU Memory:")
            print(f"  - Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"  - Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
        return True
    
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu():
    """GPU 상태 확인"""
    print("\n" + "="*50)
    print("Testing GPU...")
    print("="*50)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - cuDNN version: {torch.backends.cudnn.version()}")
        print(f"  - Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n  GPU {i}:")
            print(f"  - Name: {torch.cuda.get_device_name(i)}")
            print(f"  - Compute Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  - Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # 간단한 GPU 연산 테스트
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print(f"\n✓ GPU computation test passed")
        
        return True
    else:
        print(f"✗ CUDA is not available")
        print(f"  Training will use CPU (much slower)")
        return False


def main():
    print("\n" + "="*60)
    print(" SAM + CLIP Fine-tuning Environment Test")
    print("="*60)
    
    # GPU 테스트
    gpu_ok = test_gpu()
    
    # 데이터셋 테스트
    dataset_ok = test_dataset()
    
    # 모델 테스트
    model_ok = test_model()
    
    # 최종 결과
    print("\n" + "="*60)
    print(" Test Summary")
    print("="*60)
    print(f"GPU Test:     {'✓ PASS' if gpu_ok else '✗ FAIL (CPU will be used)'}")
    print(f"Dataset Test: {'✓ PASS' if dataset_ok else '✗ FAIL'}")
    print(f"Model Test:   {'✓ PASS' if model_ok else '✗ FAIL'}")
    
    if dataset_ok and model_ok:
        print("\n✓ All tests passed! Ready to start training.")
        print("\nTo start training, run:")
        print("  ./run_training.sh")
        print("or")
        print("  python train.py --data-root ./VehicleSeg10K --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth")
    else:
        print("\n✗ Some tests failed. Please fix the issues before training.")
    
    print("="*60)


if __name__ == '__main__':
    main()
