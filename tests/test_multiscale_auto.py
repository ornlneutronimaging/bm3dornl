"""Regression: multiscale with automatic sigma must act on flat-background data.

The multiscale path's internal noise estimator returns the minimum patch sigma,
targeting the air-region noise floor. On simulated sinograms the air is
numerically flat, so that minimum was float quantization residue (~1e-10); the
variance lock's threshold collapsed with it and the whole correction silently
returned its input. This mirrors the JOSS benchmark invocation that exposed it.
"""

import numpy as np

from bm3dornl import bm3d_ring_artifact_removal
from bm3dornl.phantom import shepp_logan_phantom, generate_sinogram


def test_multiscale_auto_sigma_acts_on_flat_background():
    phantom = shepp_logan_phantom(size=128)
    sinogram, _ = generate_sinogram(phantom, 1.0)

    # Per-column detector gain error, i.e. vertical streaks.
    rng = np.random.default_rng(42)
    gain = rng.normal(1.0, 0.02, sinogram.shape[1]).astype(np.float32)
    noisy = (sinogram * gain).astype(np.float32)

    corrected = bm3d_ring_artifact_removal(
        noisy,
        mode="streak",
        sigma_random=0.0,  # auto-estimate: the failing configuration
        multiscale=True,
        num_scales=2,
        patch_size=8,
        step_size=4,
        search_window=24,
        max_matches=32,
    )

    assert corrected.shape == noisy.shape
    # The broken code changed the output only by ~1e-10 numerical dust, so the
    # tolerance is the point of this assertion: it distinguishes a real
    # correction from dust. rtol=0 keeps the threshold purely absolute.
    assert not np.allclose(noisy, corrected, atol=1e-6, rtol=0.0), (
        "multiscale with automatic sigma made no measurable change to a "
        "flat-background sinogram; the noise-floor estimate has collapsed again"
    )

    # The smooth sample level must survive; only streaks are removable.
    mid = noisy.shape[1] // 2
    level_in = float(noisy[:, mid - 5:mid + 5].mean())
    level_out = float(corrected[:, mid - 5:mid + 5].mean())
    assert abs(level_in - level_out) < 0.05 * abs(level_in), (
        f"sample level moved: {level_in} -> {level_out}"
    )
