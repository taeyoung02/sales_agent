#!/usr/bin/env python3
import numpy as np
import os
import glob

# language_features 디렉토리 경로
features_dir = "data/kolon_sample_floating_langsplat/language_features"

if not os.path.exists(features_dir):
    print(f"Directory not found: {features_dir}")
    exit(1)

# 모든 npy 파일 찾기
npy_files = sorted(glob.glob(os.path.join(features_dir, "*.npy")))

print(f"Found {len(npy_files)} npy files\n")
print("=" * 80)

# 처음 5개 파일과 랜덤 샘플 체크
sample_files = npy_files[:80]

nan_count = 0
inf_count = 0

for i, path in enumerate(sample_files, 1):
    if os.path.exists(path):
        data = np.load(path)
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        
        print(f"\n[{i}] File: {os.path.basename(path)}")
        print(f"    Shape: {data.shape}")
        print(f"    Dtype: {data.dtype}")
        print(f"    Has NaN?: {has_nan}")
        print(f"    Has Inf?: {has_inf}")
        
        if not has_nan and not has_inf:
            print(f"    Max value: {np.max(data):.6f}")
            print(f"    Min value: {np.min(data):.6f}")
            print(f"    Mean value: {np.mean(data):.6f}")
        else:
            if has_nan:
                nan_count += 1
                nan_total = np.isnan(data).sum()
                print(f"    ⚠️  NaN count: {nan_total} / {data.size}")
            if has_inf:
                inf_count += 1
                inf_total = np.isinf(data).sum()
                print(f"    ⚠️  Inf count: {inf_total} / {data.size}")
    else:
        print(f"\n[{i}] File not found: {path}")

print("\n" + "=" * 80)
print(f"\nSummary:")
print(f"  Files checked: {len(sample_files)}")
print(f"  Files with NaN: {nan_count}")
print(f"  Files with Inf: {inf_count}")

if nan_count > 0 or inf_count > 0:
    print("\n⚠️  WARNING: Found NaN or Inf values in feature files!")
    print("   This will cause issues in CF3 training.")
else:
    print("\n✓ All checked files are clean (no NaN/Inf values)")
