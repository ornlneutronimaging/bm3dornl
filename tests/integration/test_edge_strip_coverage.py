"""Every output pixel is filtered, including the bottom and right strip.

The impulse gain d(out)/d(in) at one pixel shows this directly. An
unfiltered pixel passes an impulse through with gain 1.000. A filtered pixel
spreads it over the patches of its groups and keeps a small part of it.

The public API computes in float32: it casts 2D input with
``np.ascontiguousarray(..., dtype=np.float32)`` and 3D input with
``astype(np.float32)``. The float64 kernels are checked through the f64 bindings:
``bm3d_ring_artifact_removal_2d_f64`` for 2D, and the f64 stack kernels
(hard thresholding, then Wiener on that pilot) for 3D.
"""

import numpy as np
import pytest

from bm3dornl import bm3d_ring_artifact_removal, bm3d_rust
from bm3dornl.bm3d import DEFAULT_SINGLE_SCALE_THRESHOLD

SIGMA = 0.05
MAX_GAIN = 0.5
PATCH_SIZE, STEP_SIZE, SEARCH_WINDOW, MAX_MATCHES = 8, 4, 24, 16


def noisy_input(shape, is_stack, dtype):
    """Smooth image plus Gaussian noise; a stack holds two noise draws."""
    rows, cols = shape
    rng = np.random.default_rng(rows * 1000 + cols)
    r = np.arange(rows)[:, None]
    c = np.arange(cols)[None, :]
    clean = 0.5 + 0.15 * np.sin(2 * np.pi * r / 23.0) * np.cos(2 * np.pi * c / 31.0)
    if is_stack:
        x = np.stack([clean + rng.normal(0.0, SIGMA, clean.shape) for _ in range(2)])
    else:
        x = clean + rng.normal(0.0, SIGMA, clean.shape)
    return x.astype(dtype)


def normalised_sigma(x):
    # bm3dornl applies sigma_random to the min/max-normalised image.
    return SIGMA / float(x.max() - x.min())


def public_api(mode, **geometry):
    def run(x):
        return bm3d_ring_artifact_removal(
            x, mode=mode, sigma_random=normalised_sigma(x), **geometry
        )

    return run


def f64_2d_binding(mode, **geometry):
    def run(x):
        return bm3d_rust.bm3d_ring_artifact_removal_2d_f64(
            x, mode, sigma_random=normalised_sigma(x), **geometry
        )

    return run


def streak_psd():
    """Anisotropic PSD."""
    freq = np.arange(PATCH_SIZE)
    profile = np.exp(-0.5 * (np.minimum(freq, PATCH_SIZE - freq) / 0.6) ** 2)
    return np.tile(profile[:, None], (1, PATCH_SIZE))


def f64_stack_kernels(noise_model):
    """Both f64 stack passes on raw data (no normalisation).

    ``white`` uses a scalar noise model. ``streak`` adds the anisotropic PSD
    and a per-column sigma map, so the colored-noise branch runs.
    """

    def run(x):
        n, _, cols = x.shape
        if noise_model == "white":
            sigma_psd, sigma_map = np.zeros((1, 1)), np.zeros((1, 1, 1))
        else:
            sigma_psd, sigma_map = streak_psd(), np.full((n, 1, cols), SIGMA / 2)
        geometry = (PATCH_SIZE, STEP_SIZE, SEARCH_WINDOW, MAX_MATCHES)
        pilot = bm3d_rust.bm3d_hard_thresholding_stack_f64(
            x, x, sigma_psd, sigma_map, SIGMA, DEFAULT_SINGLE_SCALE_THRESHOLD, *geometry
        )
        return bm3d_rust.bm3d_wiener_filtering_stack_f64(
            x, pilot, sigma_psd, sigma_map, SIGMA, *geometry
        )

    return run


def impulse_gain(run, x, index):
    """d(out)/d(in) at ``index`` for a 3-sigma impulse at that pixel."""
    base = np.asarray(run(x), dtype=np.float64)
    bumped = x.copy()
    # Push toward the mean so the normalisation range stays the same.
    bumped[index] += 3 * SIGMA if x[index] < x.mean() else -3 * SIGMA
    assert bumped.max() == x.max() and bumped.min() == x.min(), (
        "the impulse changed the min/max normalisation range"
    )
    delta = float(bumped[index]) - float(x[index])
    out = np.asarray(run(bumped), dtype=np.float64)
    return (out[index] - base[index]) / delta


def gains_at(run, x, pixels):
    is_stack = x.ndim == 3
    return {
        name: impulse_gain(run, x, (0, r, c) if is_stack else (r, c))
        for name, (r, c) in pixels.items()
    }


def edge_pixels(rows, cols):
    return {
        "last row": (rows - 1, cols // 2),
        "last column": (rows // 2, cols - 1),
        "corner": (rows - 1, cols - 1),
    }


# path id -> (filter, input is a stack, input dtype)
PATHS = {
    "public-2d-generic": (public_api("generic"), False, np.float32),
    "public-2d-streak": (public_api("streak"), False, np.float32),
    "public-3d-generic": (public_api("generic"), True, np.float32),
    "public-3d-streak": (public_api("streak"), True, np.float32),
    "f64-2d-generic": (f64_2d_binding("generic"), False, np.float64),
    "f64-2d-streak": (f64_2d_binding("streak"), False, np.float64),
    "f64-stack-white": (f64_stack_kernels("white"), True, np.float64),
    "f64-stack-streak": (f64_stack_kernels("streak"), True, np.float64),
}


def assert_edge_pixels_filtered(path_spec, shape, label):
    run, is_stack, dtype = path_spec
    rows, cols = shape
    x = noisy_input(shape, is_stack, dtype)
    gains = gains_at(run, x, edge_pixels(rows, cols))
    assert all(g < MAX_GAIN for g in gains.values()), (
        f"{rows}x{cols} {label}: impulse gains {gains} reach {MAX_GAIN}; "
        f"gain 1.0 means the pixel is returned unfiltered"
    )


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    "shape",
    [(65, 81), (66, 82), (67, 83)],
    ids=["remainder1", "remainder2", "remainder3"],
)
def test_impulse_is_filtered_in_bottom_and_right_strip(shape, path):
    assert_edge_pixels_filtered(PATHS[path], shape, path)


# (66 - 7) % 3 and (81 - 7) % 3 are 2.
PATCH_7_STEP_3 = {"patch_size": 7, "step_size": 3}
PATCH_7_PATHS = {
    "public-2d-generic": (public_api("generic", **PATCH_7_STEP_3), False, np.float32),
    "public-2d-streak": (public_api("streak", **PATCH_7_STEP_3), False, np.float32),
    "public-3d-generic": (public_api("generic", **PATCH_7_STEP_3), True, np.float32),
    "public-3d-streak": (public_api("streak", **PATCH_7_STEP_3), True, np.float32),
    "f64-2d-generic": (f64_2d_binding("generic", **PATCH_7_STEP_3), False, np.float64),
    "f64-2d-streak": (f64_2d_binding("streak", **PATCH_7_STEP_3), False, np.float64),
}


@pytest.mark.parametrize("path", PATCH_7_PATHS)
@pytest.mark.parametrize("shape", [(66, 66), (66, 81)], ids=["66x66", "66x81"])
def test_impulse_is_filtered_in_strip_with_patch_7_step_3(shape, path):
    assert_edge_pixels_filtered(PATCH_7_PATHS[path], shape, f"{path} patch 7 step 3")


@pytest.mark.parametrize("path", PATHS)
def test_impulse_gain_is_small_on_aligned_shape(path):
    # Control: on 64 x 80 the plain reference grid reaches both edges, so
    # edge and interior pixels must all stay below the bound.
    run, is_stack, dtype = PATHS[path]
    rows, cols = 64, 80
    x = noisy_input((rows, cols), is_stack, dtype)
    pixels = edge_pixels(rows, cols)
    pixels["interior"] = (rows // 2, cols // 2)
    gains = gains_at(run, x, pixels)
    assert all(g < MAX_GAIN for g in gains.values()), (
        f"{rows}x{cols} {path}: impulse gains {gains} reach {MAX_GAIN}"
    )
