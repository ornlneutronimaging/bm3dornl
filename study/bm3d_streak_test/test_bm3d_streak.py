#!/usr/bin/env python
"""
Isolated test for bm3d-streak-removal package.

This script tests bm3d-streak-removal in an isolated environment with
compatible scipy version (pre-1.11 where scipy.signal.gaussian existed).

Note: Uses scikit-image for phantom generation since bm3dornl requires
Python 3.12+ which is incompatible with bm3d-streak-removal's scipy requirement.
"""

import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.data import shepp_logan_phantom
from skimage.transform import radon

# bm3d-streak-removal imports
from bm3d_streak_removal import multiscale_streak_removal, extreme_streak_attenuation

# Test parameters - use larger size for bm3d-streak-removal compatibility
# bm3d-streak-removal requires sufficient image size for BM4D block matching
PHANTOM_SIZE = 512
SCAN_STEP = 0.5  # Degrees per projection
np.random.seed(42)


def generate_test_data():
    """Generate synthetic sinogram with ring artifacts using scikit-image.

    Uses 512x512 phantom (larger than main benchmark's 256x256) to ensure
    sufficient detector pixels for bm3d-streak-removal's BM4D block matching.
    Sinogram shape: 720x729 (theta x detector).
    """
    print("Generating synthetic test data...")

    # Create Shepp-Logan phantom
    phantom = shepp_logan_phantom()
    # Resize to desired size
    from skimage.transform import resize
    phantom = resize(phantom, (PHANTOM_SIZE, PHANTOM_SIZE), anti_aliasing=True)

    # Generate sinogram via Radon transform
    # Use 360 degrees with 0.5 degree step = 720 projections (matches main benchmark)
    theta = np.arange(-180, 180, SCAN_STEP)  # 720 projections
    sinogram = radon(phantom, theta=theta)

    # Normalize to [0, 1]
    sinogram = (sinogram - sinogram.min()) / (sinogram.max() - sinogram.min())

    # Add ring artifacts (vertical stripes in sinogram)
    # Simulate detector gain variations
    n_detectors = sinogram.shape[0]
    detector_gain = np.ones(n_detectors)

    # Add random gain variations
    np.random.seed(42)
    n_bad_detectors = n_detectors // 20  # 5% bad detectors
    bad_indices = np.random.choice(n_detectors, n_bad_detectors, replace=False)
    detector_gain[bad_indices] = np.random.uniform(0.95, 1.05, n_bad_detectors)

    sinogram_with_rings = sinogram * detector_gain[:, np.newaxis]

    print(f"  Phantom shape: {phantom.shape}")
    print(f"  Sinogram shape: {sinogram.shape}")

    return sinogram.T, sinogram_with_rings.T  # Transpose to (theta, detector)


def run_bm3d_streak_removal(sinogram):
    """Apply bm3d-streak-removal.

    Uses the standard pipeline with data scaled to appropriate range.
    Expected input shape for multiscale_streak_removal: (angle, sinogram_idx, detector)
    So single sinogram of shape (N_angles, N_detectors) becomes (N_angles, 1, N_detectors).
    """
    # sinogram is (theta, detector) = (720, 512)
    # Expected shape: (angle, sino_idx, detector) = (720, 1, 512)
    sino_3d = sinogram.astype(np.float64)[:, np.newaxis, :]
    print(f"  Input shape for bm3d-streak: {sino_3d.shape}")
    # Scale to [0.1, 1.1] to avoid log(0) issues
    sino_scaled = sino_3d + 0.1
    sino_log = np.log(sino_scaled)
    sino_attenuated = extreme_streak_attenuation(sino_log)
    # Use default parameters for fair comparison
    sino_denoised = multiscale_streak_removal(sino_attenuated)
    result = np.exp(sino_denoised) - 0.1
    return result[:, 0, :]


def compute_metrics(result, ground_truth):
    """Compute PSNR and SSIM."""
    result = np.asarray(result, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)

    result_norm = (result - result.min()) / (result.max() - result.min() + 1e-10)
    gt_norm = (ground_truth - ground_truth.min()) / (
        ground_truth.max() - ground_truth.min() + 1e-10
    )

    psnr = peak_signal_noise_ratio(gt_norm, result_norm, data_range=1.0)
    ssim = structural_similarity(gt_norm, result_norm, data_range=1.0)
    return psnr, ssim


def main():
    import os
    import csv
    from pathlib import Path

    print("=" * 60)
    print("bm3d-streak-removal Isolated Test")
    print("=" * 60)

    ground_truth, sinogram_rings = generate_test_data()

    print("\nRunning bm3d-streak-removal...")
    start = time.perf_counter()
    try:
        result = run_bm3d_streak_removal(sinogram_rings)
        elapsed = time.perf_counter() - start

        psnr, ssim = compute_metrics(result, ground_truth)

        print(f"  Time: {elapsed:.3f} s")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  SSIM: {ssim:.4f}")
        print(f"  Output shape: {result.shape}")

        # Save result for comparison
        np.save("bm3d_streak_result.npy", result)
        np.save("bm3d_streak_input.npy", sinogram_rings)
        np.save("bm3d_streak_ground_truth.npy", ground_truth)
        print("\n  Results saved to .npy files")

        # Save results to linux_x86_64 folder for integration
        results_dir = Path(__file__).parent.parent / "results" / "linux_x86_64"
        results_dir.mkdir(parents=True, exist_ok=True)

        bm3d_streak_csv = results_dir / "bm3d_streak_results.csv"
        with open(bm3d_streak_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["method", "time_mean", "time_std", "psnr", "ssim", "phantom_size", "notes"])
            writer.writerow([
                "bm3d-streak-removal",
                f"{elapsed:.3f}",
                "0.000",  # Single run
                f"{psnr:.2f}",
                f"{ssim:.4f}",
                PHANTOM_SIZE,
                "Tested with 512x512 phantom (larger than main benchmark 256x256) due to BM4D block size requirements"
            ])
        print(f"  CSV saved to: {bm3d_streak_csv}")

        return {"time": elapsed, "psnr": psnr, "ssim": ssim, "success": True}

    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"time": elapsed, "error": str(e), "success": False}


if __name__ == "__main__":
    result = main()
    print("\n" + "=" * 60)
    if result["success"]:
        print("TEST PASSED")
    else:
        print("TEST FAILED")
    print("=" * 60)
