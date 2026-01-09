
import numpy as np
import time
import resource
import os
import sys
from bm3dornl.bm3d import bm3d_ring_artifact_removal

def get_peak_rss_mb():
    """Returns peak RSS in MB."""
    if sys.platform == "darwin":
        # Mac return bytes? No, getrusage on Mac returns bytes? 
        # Actually it's often KB or bytes depending on OS.
        # On Mac usually bytes? No, documentation says bytes on Mac, KB on Linux.
        # Let's check interactively or assume standard scaling.
        # We can just check start vs end.
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        return rusage.ru_maxrss / 1024 / 1024 # If bytes -> MB
        # If result is too small, it was KB.
    else:
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        return rusage.ru_maxrss / 1024

def test_memory():
    # Target size: ~500 MB input
    # Float32 = 4 bytes.
    # 500 MB = 125 M floats.
    # 512 x 512 = 262,144 pixels.
    # Frames = 125,000,000 / 262,144 ~= 476 frames.
    N = 400
    H, W = 512, 512
    
    print(f"Generating stack {N}x{H}x{W} (Float32)...")
    # 400 * 512 * 512 * 4 bytes = 400 MB.
    input_size_mb = (N * H * W * 4) / (1024 * 1024)
    print(f"Input Size: {input_size_mb:.2f} MB")
    
    # Force GC
    import gc
    gc.collect()
    
    baseline_rss = get_peak_rss_mb() # This is PEAK so far. Current RSS might be lower.
    # We want Current RSS.
    # resource.getrusage gives maxrss over lifetime.
    # We can measure current RSS via psutil if available, or just rely on Peak.
    # If we allocate input, Peak increases.
    # Then processing, Peak increases more.
    
    stack = np.random.normal(0, 1, (N, H, W)).astype(np.float32)
    
    rss_after_alloc = get_peak_rss_mb()
    print(f"RSS after allocation: {rss_after_alloc:.2f} MB")
    
    print("Running BM3D...")
    start_time = time.time()
    # Use stack mode
    denoised = bm3d_ring_artifact_removal(stack, mode="streak", sigma_random=0.1, patch_size=8, max_matches=32)
    end_time = time.time()
    
    rss_peak = get_peak_rss_mb()
    print(f"RSS Peak: {rss_peak:.2f} MB")
    print(f"Expansion: {(rss_peak - rss_after_alloc) / input_size_mb:.2f}x (Processing Overhead relative to Input)")
    print(f"Total / Input: {rss_peak / input_size_mb:.2f}x")
    
    print(f"Time: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    test_memory()
