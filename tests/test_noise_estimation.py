"""Contract tests for estimate_noise_sigma (issue #134).

The estimator measures the amplitude of vertical streaks — the sinogram
signature of ring artifacts — not pixel-level i.i.d. noise: its vertical
Gaussian pre-filter deliberately suppresses pixel-level noise before the
horizontal high-pass isolates column-to-column variation. Issue #134 fed it
i.i.d. noise expecting the pixel sigma back; these tests pin both sides of
the actual contract so the docstring and the behaviour cannot drift apart.
"""

import numpy as np
import pytest

from bm3dornl.utils import estimate_noise_sigma


def _streaked(streak_sigma: float, seed: int, dtype=np.float32) -> np.ndarray:
    """Constant image plus a per-column offset: pure vertical streaks."""
    rng = np.random.default_rng(seed)
    clean = np.ones((256, 512), dtype=np.float64)
    streaks = rng.normal(0.0, streak_sigma, size=(1, 512))
    return (clean + streaks).astype(dtype)


def test_recovers_vertical_streak_sigma():
    for seed in range(5):
        sigma = estimate_noise_sigma(_streaked(0.2, seed))
        assert 0.15 < sigma < 0.25, (
            f"streak sigma 0.2 estimated as {sigma} (seed {seed})"
        )


def test_iid_pixel_noise_is_suppressed_by_design():
    # The exact construction from issue #134. The vertical pre-filter
    # suppresses i.i.d. pixel noise, so the result must land far below the
    # pixel-level sigma of 0.1 (measured: ~0.013 on this input).
    np.random.seed(100)
    sinogram = np.random.randn(256, 512).astype(np.float32) * 0.1
    sigma = estimate_noise_sigma(sinogram)
    assert 0.0 < sigma < 0.05, (
        f"i.i.d. pixel noise (sigma 0.1) reported as {sigma}; the estimator "
        "should suppress it, not measure it"
    )


def test_f32_f64_parity():
    s32 = estimate_noise_sigma(_streaked(0.2, 7, np.float32))
    s64 = estimate_noise_sigma(_streaked(0.2, 7, np.float64))
    assert s64 == pytest.approx(s32, rel=1e-2)


def test_other_dtypes_convert_to_f32():
    scaled = (_streaked(0.2, 3) * 1000.0).astype(np.int32)
    sigma_int = estimate_noise_sigma(scaled)
    sigma_f32 = estimate_noise_sigma(scaled.astype(np.float32))
    assert sigma_int == pytest.approx(sigma_f32, rel=1e-6)


def test_rejects_non_2d_input():
    with pytest.raises(ValueError):
        estimate_noise_sigma(np.zeros((4, 4, 4), dtype=np.float32))
