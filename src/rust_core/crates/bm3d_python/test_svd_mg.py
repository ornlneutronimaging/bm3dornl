
import numpy as np
import bm3d_rust
import time

def create_phantom(shape=(256, 256)):
    """Create a phantom with vertical streaks."""
    img = np.zeros(shape, dtype=np.float32)
    # Add some structure
    img[50:200, 50:200] = 0.5
    # Add vertical streaks
    streaks = np.zeros(shape, dtype=np.float32)
    streaks[:, 100] = 0.2
    streaks[:, 150] = -0.1
    # Add noise
    noise = np.random.normal(0, 0.05, shape).astype(np.float32)
    
    noisy = img + streaks + noise
    return noisy

def test_svd_mg():
    print("Testing SVD-MG integration...")
    img = create_phantom()
    
    # Test 1: Standalone SVD-MG
    print("Running svd_mg_removal_py...")
    start = time.time()
    res1 = bm3d_rust.svd_mg_removal_rust(img, fft_alpha=1.0, notch_width=2.0)
    end = time.time()
    print(f"Standalone done in {end-start:.4f}s. Shape: {res1.shape}")
    
    # Test 2: Pipeline SVD-MG
    print("Running bm3d_ring_artifact_removal_2d(mode='svd_mg')...")
    start = time.time()
    res2 = bm3d_rust.bm3d_ring_artifact_removal_2d(
        img, 
        mode='svd_mg',
        fft_alpha=2.0,
        notch_width=1.0
    )
    end = time.time()
    print(f"Pipeline done in {end-start:.4f}s. Shape: {res2.shape}")
    
    # Verification:
    # 1. Output should not be identical to input (something happened)
    diff = np.abs(img - res2).mean()
    print(f"Mean Abs Diff (Input vs Output): {diff:.6f}")
    if diff < 1e-6:
        print("WARNING: Output seems identical to input!")
    else:
        print("SUCCESS: Image modified.")

    print("\nTest Complete.")

if __name__ == "__main__":
    test_svd_mg()
