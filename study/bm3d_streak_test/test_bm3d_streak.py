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

# Test parameters
PHANTOM_SIZE = 256
np.random.seed(42)


def generate_test_data():
    """Generate synthetic sinogram with ring artifacts using scikit-image."""
    print("Generating synthetic test data...")

    # Create Shepp-Logan phantom
    phantom = shepp_logan_phantom()
    # Resize to desired size
    from skimage.transform import resize
    phantom = resize(phantom, (PHANTOM_SIZE, PHANTOM_SIZE), anti_aliasing=True)

    # Generate sinogram via Radon transform
    theta = np.linspace(0., 180., PHANTOM_SIZE, endpoint=False)
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
    """Apply bm3d-streak-removal."""
    sino_3d = sinogram.astype(np.float64)[np.newaxis, :, :]
    sino_log = np.log1p(sino_3d)
    sino_attenuated = extreme_streak_attenuation(sino_log)
    sino_denoised = multiscale_streak_removal(sino_attenuated)
    result = np.expm1(sino_denoised)
    return result[0]


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
