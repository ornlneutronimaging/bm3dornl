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
        sigma_random=0.0,  # auto-estimate
        multiscale=True,
        num_scales=2,
        patch_size=8,
        step_size=4,
        search_window=24,
        max_matches=32,
    )

    assert corrected.shape == noisy.shape
    assert not np.allclose(noisy, corrected, atol=1e-6, rtol=0.0), (
        "multiscale with automatic sigma made no measurable change to a "
        "flat-background sinogram; the noise-floor estimate has collapsed again"
    )

    mid = noisy.shape[1] // 2
    level_in = float(noisy[:, mid - 5 : mid + 5].mean())
    level_out = float(corrected[:, mid - 5 : mid + 5].mean())
    assert abs(level_in - level_out) < 0.05 * abs(level_in), (
        f"sample level moved: {level_in} -> {level_out}"
    )


def test_log_domain_input_flag_is_a_pure_change_of_domain():
    """Feeding pre-logged data with log_domain_input=True must equal feeding
    linear data with the default, up to the exp round-trip.

    Tolerance: numpy's and Rust's f32 ln differ by an ulp, and BM3D's hard
    threshold is discontinuous, so single-ulp input differences flip a few
    coefficients."""
    rng = np.random.default_rng(7)
    sinogram = (0.3 + 0.5 * rng.random((96, 200))).astype(np.float32)

    via_default = bm3d_ring_artifact_removal(
        sinogram, mode="streak", sigma_random=0.0, multiscale=True, num_scales=2
    )
    via_flag = np.exp(
        bm3d_ring_artifact_removal(
            np.log(sinogram),
            mode="streak",
            sigma_random=0.0,
            multiscale=True,
            num_scales=2,
            log_domain_input=True,
        )
    )
    np.testing.assert_allclose(via_flag, via_default, rtol=0, atol=1e-2)
