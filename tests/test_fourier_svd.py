import numpy as np
import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


def test_fourier_svd_import():
    try:
        from bm3dornl import fourier_svd  # noqa: F401
    except ImportError:
        pytest.fail("Could not import bm3dornl.fourier_svd module")


def test_fourier_svd_execution():
    from bm3dornl import fourier_svd

    noise_sigma = 1.0
    img = np.random.normal(100, noise_sigma, (100, 100)).astype(np.float32)

    streak_col = 50
    streak_val = 0.2 * noise_sigma
    img[:, streak_col] += streak_val

    corrected = fourier_svd.fourier_svd_removal(img)

    assert corrected.shape == img.shape, "Output shape mismatch"
    assert corrected.dtype == img.dtype, "Output dtype mismatch"

    streak_mean = np.mean(corrected[:, streak_col])
    neighbor_mean = np.mean(corrected[:, streak_col - 1])

    diff = abs(streak_mean - neighbor_mean)
    print(f"Subtle Streak Removal: Orig {streak_val}, Residual {diff}")
    assert diff < 1.0, f"Streak not removed effectively. Residual diff: {diff}"


def test_fourier_svd_flat_background_sinogram():
    from bm3dornl import fourier_svd
    from bm3dornl.phantom import shepp_logan_phantom, generate_sinogram

    phantom = shepp_logan_phantom(size=128)
    sinogram, _ = generate_sinogram(phantom, 1.0)

    # Per-column detector gain error, i.e. vertical streaks.
    rng = np.random.default_rng(42)
    gain = rng.normal(1.0, 0.02, sinogram.shape[1]).astype(np.float32)
    noisy = (sinogram * gain).astype(np.float32)

    corrected = fourier_svd.fourier_svd_removal(noisy)

    assert corrected.shape == noisy.shape
    assert not np.allclose(noisy, corrected, atol=1e-6), (
        "fourier_svd_removal made no measurable change to a flat-background "
        "sinogram; the magnitude gate has collapsed again"
    )

    without_fft = fourier_svd.fourier_svd_removal(noisy, fft_alpha=0.0)
    assert not np.allclose(corrected, without_fft, atol=1e-6), (
        "fft_alpha has no influence on the result"
    )


def test_fourier_svd_wall_protection():
    from bm3dornl import fourier_svd

    img = np.zeros((100, 100), dtype=np.float32)
    img[:, 40:60] = 100.0

    img += np.random.normal(0, 1, (100, 100))

    corrected = fourier_svd.fourier_svd_removal(img)

    wall_mean = np.mean(corrected[:, 50])
    print(f"Wall Protection: Mean {wall_mean}")
    assert wall_mean > 90.0, f"Wall attacked! Mean dropped to {wall_mean}"
