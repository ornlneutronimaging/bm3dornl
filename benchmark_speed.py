import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from bm3dornl import bm3d
import numpy as np
import time


def run_benchmark():
    print("Generating synthetic data (512x512)...")
    np.random.seed(42)
    img = np.random.rand(512, 512).astype(np.float32)

    print("Warming up JIT/cache...")
    bm3d.bm3d_ring_artifact_removal(img[:64, :64], mode="streak")

    print("\n--- Benchmarking Single Scale ---")
    start_time = time.time()
    _ = bm3d.bm3d_ring_artifact_removal(img, mode="streak")
    end_time = time.time()
    single_time = end_time - start_time
    print(f"Single Scale Time: {single_time:.4f} s")

    print("\n--- Benchmarking Multi Scale ---")
    start_time = time.time()
    _ = bm3d.bm3d_ring_artifact_removal(img, mode="streak", multiscale=True)
    end_time = time.time()
    multi_time = end_time - start_time
    print(f"Multi Scale Time:  {multi_time:.4f} s")

    print(f"\nRatio (Multi/Single): {multi_time / single_time:.2f}x (Slower)")


if __name__ == "__main__":
    run_benchmark()
