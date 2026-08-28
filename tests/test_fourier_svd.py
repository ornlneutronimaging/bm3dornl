import numpy as np
import pytest
import os
import sys

# Ensure we can import bm3dornl
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

def test_fourier_svd_import():
    try:
        from bm3dornl import fourier_svd
    except ImportError:
        pytest.fail("Could not import bm3dornl.fourier_svd module")

def test_fourier_svd_execution():
    from bm3dornl import fourier_svd

    # Create synthetic sinogram with a SUBTLE streak (Operating Range)
    # The algorithm protects High Contrast features (Walls).
    # Streaks must be < ~3 * NoiseSigma to be removed.
    noise_sigma = 1.0
    img = np.random.normal(100, noise_sigma, (100, 100)).astype(np.float32)

    # Add a streak at column 50
    streak_col = 50
    streak_val = 0.2 * noise_sigma # 0.2 sigma (Ring Artifact Regime). SVD Gain ~10x -> SNR 2.0. Steps < 3.0.
    img[:, streak_col] += streak_val

    # Run Fourier-SVD
    corrected = fourier_svd.fourier_svd_removal(img)

    assert corrected.shape == img.shape, "Output shape mismatch"
    assert corrected.dtype == img.dtype, "Output dtype mismatch"

    # Check streak removal
    # The mean of the streak column should be reduced towards the neighbor mean
    streak_mean = np.mean(corrected[:, streak_col])
    neighbor_mean = np.mean(corrected[:, streak_col-1])

    diff = abs(streak_mean - neighbor_mean)
    # Original diff was 2.5. After correction should be < 1.0
    print(f"Subtle Streak Removal: Orig {streak_val}, Residual {diff}")
    assert diff < 1.0, f"Streak not removed effectively. Residual diff: {diff}"

def test_fourier_svd_flat_background_sinogram():
    """Regression for issue #133.

    A sinogram surrounded by air has a column profile whose majority is flat.
    MAD is the median of the absolute deviations, so it collapsed to ~1e-11
    there, the magnitude gate's threshold collapsed with it, and every column
    carrying real streak signal was suppressed. The result was an exact no-op
    that no parameter value could influence.
    """
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
    # The tolerance is the point of this assertion: the broken code did not
    # return bitwise-identical output, it changed values by ~1e-9 numerical
    # dust. Exact-inequality would therefore pass on the bug; requiring a
    # change above 1e-6 is what distinguishes a real correction from dust.
    assert not np.allclose(noisy, corrected, atol=1e-6), (
        "fourier_svd_removal made no measurable change to a flat-background "
        "sinogram; the magnitude gate has collapsed again"
    )

    # The correction must also respond to its parameters. Before the fix every
    # setting returned the same sub-epsilon result.
    without_fft = fourier_svd.fourier_svd_removal(noisy, fft_alpha=0.0)
    assert not np.allclose(corrected, without_fft, atol=1e-6), (
        "fft_alpha has no influence on the result"
    )


def test_fourier_svd_wall_protection():
    from bm3dornl import fourier_svd

    # Create image with a "Wall" (Wide structure)
    # A wide block [40:60] with high intensity
    img = np.zeros((100, 100), dtype=np.float32)
    img[:, 40:60] = 100.0

    # Add noise
    img += np.random.normal(0, 1, (100, 100))

    # Run Fourier-SVD
    corrected = fourier_svd.fourier_svd_removal(img)

    # Check that wall is preserved
    # Mean of wall center should still be close to 100
    wall_mean = np.mean(corrected[:, 50])
    print(f"Wall Protection: Mean {wall_mean}")
    assert wall_mean > 90.0, f"Wall attacked! Mean dropped to {wall_mean}"
