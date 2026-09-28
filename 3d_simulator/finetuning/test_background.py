"""
Test script to verify background mask creation
"""
import numpy as np
from dataset import VehicleSegDataset

# Create dataset
dataset = VehicleSegDataset(
    root_dir='./VehicleSeg10K',
    split='train'
)

print(f"Total samples: {len(dataset)}")
print(f"Labels: {VehicleSegDataset.LABELS}")
print(f"Number of classes: {len(VehicleSegDataset.LABELS)}")

# Test first sample
sample = dataset[0]

print(f"\n=== Sample 0 ===")
print(f"Image path: {sample['image_path']}")
print(f"Number of masks: {len(sample['masks'])}")
print(f"Labels: {sample['labels']}")
print(f"Label indices: {sample['label_indices']}")

# Check if background "Others" is included
others_count = sum(1 for label in sample['labels'] if label == "Others")
print(f"\nBackground 'Others' masks: {others_count}")

# Check mask coverage
if len(sample['masks']) > 0:
    h, w = sample['masks'][0].shape
    total_area = h * w
    
    # Sum all mask areas
    covered_area = 0
    for mask in sample['masks']:
        covered_area += mask.sum()
    
    coverage = (covered_area / total_area) * 100
    print(f"\nTotal image area: {total_area}")
    print(f"Covered area: {covered_area}")
    print(f"Coverage: {coverage:.2f}%")
    
    # Check for overlaps
    combined_mask = np.zeros((h, w), dtype=np.float32)
    for mask in sample['masks']:
        combined_mask += mask
    
    overlap_area = (combined_mask > 1).sum()
    print(f"Overlap area: {overlap_area} pixels ({(overlap_area/total_area)*100:.2f}%)")
    
    # Verify background mask
    for i, (mask, label) in enumerate(zip(sample['masks'], sample['labels'])):
        if label == "Others":
            print(f"\nBackground mask #{i}:")
            print(f"  Area: {mask.sum()} pixels ({(mask.sum()/total_area)*100:.2f}%)")
            print(f"  Shape: {mask.shape}")

print("\n✓ Test completed")
