import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from bm3dornl import bm3d
import numpy as np
import time


def run_benchmark():
    print("Generating synthetic data (512x512)...")
    np.random.seed(42)
    # 512x512 float32 image
    img = np.random.rand(512, 512).astype(np.float32)

    # Warmup
    print("Warming up JIT/cache...")
    bm3d.bm3d_ring_artifact_removal(img[:64, :64], mode="streak")

    print("\n--- Benchmarking Single Scale ---")
    start_time = time.time()
    # Using 'streak' mode to trigger core logic likely used in multiscale?
    # Actually multiscale is a specific flag.
    # Let's just run standard bm3d first as baseline.
    _ = bm3d.bm3d_ring_artifact_removal(img, mode="streak")
    end_time = time.time()
    single_time = end_time - start_time
    print(f"Single Scale Time: {single_time:.4f} s")

    print("\n--- Benchmarking Multi Scale ---")
    start_time = time.time()
    # 'multiscale=True' triggers the rust core multiscale logic
    _ = bm3d.bm3d_ring_artifact_removal(img, mode="streak", multiscale=True)
    end_time = time.time()
    multi_time = end_time - start_time
    print(f"Multi Scale Time:  {multi_time:.4f} s")

    print(f"\nRatio (Multi/Single): {multi_time / single_time:.2f}x (Slower)")


if __name__ == "__main__":
    run_benchmark()
