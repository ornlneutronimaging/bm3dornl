"""``step_size`` must not exceed ``patch_size`` (issue #166).

Reference patches sit ``step_size`` apart. With a step larger than the patch
they leave gaps, and the pixels in the gaps came back unfiltered (64 x 80,
patch 8, step 12: 2880 of 5120 pixels unchanged in the kernel). Every entry
point now refuses such a step with ValueError "step_size (s) must not exceed
patch_size (p)". A step equal to ``patch_size`` stays valid and filters every
pixel. A ``step_size`` of 0 raises ValueError instead of a panic.

A Rust panic reaches Python as ``pyo3_runtime.PanicException``, which
derives from BaseException, so ``pytest.raises(ValueError)`` does not catch
it; each test also checks that the type is exactly ValueError.
"""

import re

import numpy as np
import pytest

from bm3dornl import bm3d_ring_artifact_removal, bm3d_rust

SIGMA = 0.05
# (patch_size, step_size) with the step larger than the patch.
TOO_LARGE = [(8, 9), (8, 12), (7, 8)]
# The public API and the kernel return an unfiltered pixel unchanged, up to
# the normalisation round trip of the public API (below 1e-6 here). A
# filtered pixel moves by about SIGMA.
UNFILTERED_TOLERANCE = 1e-6


def noisy_step_edges(shape, dtype=np.float32, seed=7):
    """Positive step edges off the patch grid plus Gaussian noise."""
    rows, cols = shape[-2:]
    truth = np.full((rows, cols), 0.25)
    truth[:, 41:] = 0.75
    truth[35:, :] += 0.15
    noise = np.random.default_rng(seed).normal(0.0, SIGMA, shape)
    return (truth + noise).astype(dtype)


def case_id(case):
    patch_size, step_size = case
    return f"patch{patch_size}-step{step_size}"


def assert_raises_exactly(call, pattern):
    with pytest.raises(ValueError, match=pattern) as info:
        call()
    assert info.type is ValueError


def too_large_pattern(patch_size, step_size):
    return re.escape(f"step_size ({step_size}) must not exceed patch_size ({patch_size})")


def public_call(image, **kwargs):
    return np.asarray(bm3d_ring_artifact_removal(image, sigma_random=SIGMA, **kwargs))


# --- Public API ----------------------------------------------------------------

# name -> (input shape, keyword arguments)
PUBLIC = {
    "2d-generic": ((64, 80), {"mode": "generic"}),
    "2d-streak": ((64, 80), {"mode": "streak"}),
    "2d-fourier_svd": ((64, 80), {"mode": "fourier_svd"}),
    "2d-multiscale": ((64, 160), {"mode": "streak", "multiscale": True}),
    "3d-generic": ((2, 64, 80), {"mode": "generic"}),
    "3d-streak": ((2, 64, 80), {"mode": "streak"}),
    "3d-multiscale": ((2, 64, 160), {"mode": "streak", "multiscale": True}),
}


@pytest.mark.parametrize("case", TOO_LARGE, ids=case_id)
@pytest.mark.parametrize("entry", PUBLIC)
def test_public_api_refuses_step_size_above_patch_size(entry, case):
    shape, kwargs = PUBLIC[entry]
    patch_size, step_size = case
    image = noisy_step_edges(shape)
    assert_raises_exactly(
        lambda: public_call(image, patch_size=patch_size, step_size=step_size, **kwargs),
        too_large_pattern(patch_size, step_size),
    )


@pytest.mark.parametrize("patch_size", [8, 7])
@pytest.mark.parametrize("entry", PUBLIC)
def test_public_api_accepts_step_size_equal_to_patch_size(entry, patch_size):
    shape, kwargs = PUBLIC[entry]
    image = noisy_step_edges(shape)
    out = public_call(image, patch_size=patch_size, step_size=patch_size, **kwargs)
    assert out.shape == image.shape
    assert np.isfinite(out).all()


@pytest.mark.parametrize("patch_size", [8, 7])
@pytest.mark.parametrize("entry", ["2d-generic", "3d-generic"])
def test_public_generic_step_size_equal_to_patch_size_filters_every_pixel(entry, patch_size):
    # Generic mode has no streak pre-subtraction, so an unfiltered pixel
    # comes back as its input.
    shape, kwargs = PUBLIC[entry]
    image = noisy_step_edges(shape)
    out = public_call(image, patch_size=patch_size, step_size=patch_size, **kwargs)
    unfiltered = np.abs(out - image) < UNFILTERED_TOLERANCE
    assert not unfiltered.any(), f"{int(unfiltered.sum())} pixels came back unfiltered"


@pytest.mark.parametrize("entry", PUBLIC)
def test_public_api_rejects_zero_step_size(entry):
    shape, kwargs = PUBLIC[entry]
    image = noisy_step_edges(shape)
    assert_raises_exactly(lambda: public_call(image, step_size=0, **kwargs), "step_size")


# --- Bindings --------------------------------------------------------------------

HIGH_LEVEL_BINDINGS = [
    ("bm3d_ring_artifact_removal_2d", np.float32),
    ("bm3d_ring_artifact_removal_2d_f64", np.float64),
]
MULTISCALE_BINDINGS = [
    ("multiscale_bm3d_streak_removal_2d", np.float32),
    ("multiscale_bm3d_streak_removal_2d_f64", np.float64),
]
KERNEL_BINDINGS = [
    ("bm3d_hard_thresholding", np.float32),
    ("bm3d_wiener_filtering", np.float32),
    ("bm3d_hard_thresholding_stack", np.float32),
    ("bm3d_wiener_filtering_stack", np.float32),
    ("bm3d_hard_thresholding_f64", np.float64),
    ("bm3d_wiener_filtering_f64", np.float64),
    ("bm3d_hard_thresholding_stack_f64", np.float64),
    ("bm3d_wiener_filtering_stack_f64", np.float64),
]


def binding_ids(bindings):
    return [name for name, _ in bindings]


def kernel_call(name, dtype, patch_size, step_size):
    """Call a low-level kernel binding; a stack binding gets two slices."""
    fn = getattr(bm3d_rust, name)
    stack = "stack" in name
    image = noisy_step_edges((2, 64, 80) if stack else (64, 80), dtype=dtype)
    sigma_psd = np.zeros((1, 1), dtype=dtype)
    sigma_map = np.zeros((1, 1, 1) if stack else (1, 1), dtype=dtype)
    args = [image, image, sigma_psd, sigma_map, SIGMA]
    if "hard_thresholding" in name:
        args.append(2.7)
    args += [patch_size, step_size, 24, 16]
    return image, np.asarray(fn(*args))


@pytest.mark.parametrize("case", TOO_LARGE, ids=case_id)
@pytest.mark.parametrize("mode", ["generic", "streak", "fourier_svd"])
@pytest.mark.parametrize(
    "name,dtype", HIGH_LEVEL_BINDINGS, ids=binding_ids(HIGH_LEVEL_BINDINGS)
)
def test_high_level_binding_refuses_step_size_above_patch_size(name, dtype, mode, case):
    patch_size, step_size = case
    image = noisy_step_edges((64, 80), dtype=dtype)
    fn = getattr(bm3d_rust, name)
    assert_raises_exactly(
        lambda: fn(image, mode, sigma_random=SIGMA, patch_size=patch_size, step_size=step_size),
        too_large_pattern(patch_size, step_size),
    )


@pytest.mark.parametrize("case", TOO_LARGE, ids=case_id)
@pytest.mark.parametrize(
    "name,dtype", MULTISCALE_BINDINGS, ids=binding_ids(MULTISCALE_BINDINGS)
)
def test_multiscale_binding_refuses_step_size_above_patch_size(name, dtype, case):
    patch_size, step_size = case
    image = noisy_step_edges((64, 160), dtype=dtype)
    fn = getattr(bm3d_rust, name)
    assert_raises_exactly(
        lambda: fn(image, patch_size=patch_size, step_size=step_size),
        too_large_pattern(patch_size, step_size),
    )


@pytest.mark.parametrize("case", TOO_LARGE, ids=case_id)
@pytest.mark.parametrize("name,dtype", KERNEL_BINDINGS, ids=binding_ids(KERNEL_BINDINGS))
def test_kernel_binding_refuses_step_size_above_patch_size(name, dtype, case):
    patch_size, step_size = case
    assert_raises_exactly(
        lambda: kernel_call(name, dtype, patch_size, step_size),
        too_large_pattern(patch_size, step_size),
    )


@pytest.mark.parametrize("patch_size", [8, 7])
@pytest.mark.parametrize("name,dtype", KERNEL_BINDINGS, ids=binding_ids(KERNEL_BINDINGS))
def test_kernel_binding_step_size_equal_to_patch_size_filters_every_pixel(
    name, dtype, patch_size
):
    # The kernel returns the input value where no reference patch covers a pixel.
    image, out = kernel_call(name, dtype, patch_size, patch_size)
    assert out.shape == image.shape
    unchanged = out == image
    assert not unchanged.any(), f"{int(unchanged.sum())} pixels came back unfiltered"


@pytest.mark.parametrize("name,dtype", KERNEL_BINDINGS, ids=binding_ids(KERNEL_BINDINGS))
def test_kernel_binding_rejects_zero_step_size(name, dtype):
    # A stride of 0 must not reach the reference grid, where it would panic.
    assert_raises_exactly(lambda: kernel_call(name, dtype, 8, 0), "step_size")
