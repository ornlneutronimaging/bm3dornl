"""Images smaller than ``patch_size`` raise ValueError on every path (issue #165).

The 2D path refuses such an image with "Image size (rows, cols) is smaller
than patch_size p". The 3D stack path, the low-level kernel bindings and the
block-matching debug bindings (``test_block_matching_rust``) used to index
past the image in block matching and panic. A Rust panic reaches
Python as ``pyo3_runtime.PanicException``, which derives from BaseException,
not from Exception, so ``except Exception`` does not catch it. Every entry
point must raise the same ValueError instead. Multiscale processing must
raise it too, also for an image that is only too short (7 x 200), which it
used to return without filtering. An image exactly ``patch_size`` in a
dimension is valid.

``pytest.raises(ValueError)`` does not catch a PanicException, so a panic
fails these tests; each test also checks that the type is exactly ValueError.
"""

import re

import numpy as np
import pytest

from bm3dornl import bm3d_ring_artifact_removal, bm3d_rust

SIGMA = 0.05
PATCH_SIZE = 8
TOO_SMALL = [(6, 64), (64, 6), (7, 7)]
# Multiscale bins horizontally, so also check a short and a narrow image
# that are wide or tall enough for binned scales.
MULTISCALE_TOO_SMALL = TOO_SMALL + [(7, 200), (200, 7)]
AT_PATCH_SIZE = [(8, 64), (64, 8), (8, 8)]
MULTISCALE_AT_PATCH_SIZE = [(8, 200), (200, 8)]


def noisy_image(shape, dtype=np.float32, seed=3):
    """Positive step image plus Gaussian noise; multiscale needs positive data."""
    rows, cols = shape[-2:]
    truth = np.full((rows, cols), 0.4)
    truth[:, cols // 2 :] = 0.75
    noise = np.random.default_rng(seed).normal(0.0, SIGMA, shape)
    return (truth + noise).astype(dtype)


def shape_id(shape):
    return "x".join(map(str, shape))


def assert_too_small(call, shape, patch_size=PATCH_SIZE):
    rows, cols = shape[-2:]
    message = f"Image size ({rows}, {cols}) is smaller than patch_size {patch_size}"
    with pytest.raises(ValueError, match=re.escape(message)) as info:
        call()
    assert info.type is ValueError


def public_call(image, **kwargs):
    return np.asarray(bm3d_ring_artifact_removal(image, sigma_random=SIGMA, **kwargs))


# --- Public API ----------------------------------------------------------------

# name -> (ndim, keyword arguments)
PUBLIC = {
    "2d-generic": (2, {"mode": "generic"}),
    "2d-streak": (2, {"mode": "streak"}),
    "2d-fourier_svd": (2, {"mode": "fourier_svd"}),
    "3d-generic": (3, {"mode": "generic"}),
    "3d-streak": (3, {"mode": "streak"}),
}
MULTISCALE = {
    "2d-multiscale": (2, {"mode": "streak", "multiscale": True}),
    "2d-multiscale-log-domain": (
        2,
        {"mode": "streak", "multiscale": True, "log_domain_input": True},
    ),
    "3d-multiscale": (3, {"mode": "streak", "multiscale": True}),
}


def entry_shape(ndim, shape):
    return (2, *shape) if ndim == 3 else shape


@pytest.mark.parametrize("shape", TOO_SMALL, ids=shape_id)
@pytest.mark.parametrize("entry", PUBLIC)
def test_public_api_rejects_images_smaller_than_patch_size(entry, shape):
    ndim, kwargs = PUBLIC[entry]
    image = noisy_image(entry_shape(ndim, shape))
    assert_too_small(lambda: public_call(image, **kwargs), shape)


@pytest.mark.parametrize("shape", MULTISCALE_TOO_SMALL, ids=shape_id)
@pytest.mark.parametrize("entry", MULTISCALE)
def test_multiscale_rejects_images_smaller_than_patch_size(entry, shape):
    ndim, kwargs = MULTISCALE[entry]
    image = noisy_image(entry_shape(ndim, shape))
    assert_too_small(lambda: public_call(image, **kwargs), shape)


@pytest.mark.parametrize("entry", ["2d-generic", "3d-generic", "3d-streak"])
def test_public_api_check_uses_the_given_patch_size(entry):
    # 8 x 64 is valid at patch_size 8 but too small at patch_size 9.
    ndim, kwargs = PUBLIC[entry]
    image = noisy_image(entry_shape(ndim, (8, 64)))
    assert_too_small(lambda: public_call(image, patch_size=9, **kwargs), (8, 64), 9)


@pytest.mark.parametrize("shape", AT_PATCH_SIZE, ids=shape_id)
@pytest.mark.parametrize("entry", PUBLIC)
def test_public_api_accepts_images_exactly_patch_size(entry, shape):
    ndim, kwargs = PUBLIC[entry]
    image = noisy_image(entry_shape(ndim, shape))
    out = public_call(image, **kwargs)
    assert out.shape == image.shape
    assert np.isfinite(out).all()


@pytest.mark.parametrize("shape", MULTISCALE_AT_PATCH_SIZE, ids=shape_id)
@pytest.mark.parametrize("entry", MULTISCALE)
def test_multiscale_accepts_images_exactly_patch_size(entry, shape):
    ndim, kwargs = MULTISCALE[entry]
    image = noisy_image(entry_shape(ndim, shape))
    out = public_call(image, **kwargs)
    assert out.shape == image.shape
    assert np.isfinite(out).all()


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


def kernel_call(name, dtype, shape, patch_size=PATCH_SIZE):
    """Call a low-level kernel binding; a stack binding gets two slices."""
    fn = getattr(bm3d_rust, name)
    stack = "stack" in name
    image = noisy_image((2, *shape) if stack else shape, dtype=dtype)
    sigma_psd = np.zeros((1, 1), dtype=dtype)
    sigma_map = np.zeros((1, 1, 1) if stack else (1, 1), dtype=dtype)
    args = [image, image, sigma_psd, sigma_map, SIGMA]
    if "hard_thresholding" in name:
        args.append(2.7)
    args += [patch_size, 4, 24, 16]
    return image, np.asarray(fn(*args))


@pytest.mark.parametrize("shape", TOO_SMALL, ids=shape_id)
@pytest.mark.parametrize("mode", ["generic", "streak", "fourier_svd"])
@pytest.mark.parametrize(
    "name,dtype", HIGH_LEVEL_BINDINGS, ids=binding_ids(HIGH_LEVEL_BINDINGS)
)
def test_high_level_binding_rejects_images_smaller_than_patch_size(name, dtype, mode, shape):
    image = noisy_image(shape, dtype=dtype)
    fn = getattr(bm3d_rust, name)
    assert_too_small(lambda: fn(image, mode, sigma_random=SIGMA), shape)


@pytest.mark.parametrize("shape", MULTISCALE_TOO_SMALL, ids=shape_id)
@pytest.mark.parametrize(
    "name,dtype", MULTISCALE_BINDINGS, ids=binding_ids(MULTISCALE_BINDINGS)
)
def test_multiscale_binding_rejects_images_smaller_than_patch_size(name, dtype, shape):
    image = noisy_image(shape, dtype=dtype)
    fn = getattr(bm3d_rust, name)
    assert_too_small(lambda: fn(image), shape)


@pytest.mark.parametrize("shape", TOO_SMALL, ids=shape_id)
@pytest.mark.parametrize("name,dtype", KERNEL_BINDINGS, ids=binding_ids(KERNEL_BINDINGS))
def test_kernel_binding_rejects_images_smaller_than_patch_size(name, dtype, shape):
    assert_too_small(lambda: kernel_call(name, dtype, shape), shape)


@pytest.mark.parametrize("name,dtype", KERNEL_BINDINGS, ids=binding_ids(KERNEL_BINDINGS))
def test_kernel_binding_check_uses_the_given_patch_size(name, dtype):
    assert_too_small(lambda: kernel_call(name, dtype, (8, 64), patch_size=9), (8, 64), 9)


@pytest.mark.parametrize("shape", AT_PATCH_SIZE, ids=shape_id)
@pytest.mark.parametrize("name,dtype", KERNEL_BINDINGS, ids=binding_ids(KERNEL_BINDINGS))
def test_kernel_binding_accepts_images_exactly_patch_size(name, dtype, shape):
    image, out = kernel_call(name, dtype, shape)
    assert out.shape == image.shape
    assert out.dtype == dtype
    assert np.isfinite(out).all()


DEBUG_BLOCK_MATCHING_BINDINGS = [
    ("test_block_matching_rust", np.float32),
    ("test_block_matching_rust_f64", np.float64),
]


def block_matching_call(name, dtype, shape):
    """Match the patch at (0, 0): patch_size 8, search window 16, 16 matches."""
    image = noisy_image(shape, dtype=dtype)
    return getattr(bm3d_rust, name)(image, 0, 0, PATCH_SIZE, 16, 16)


@pytest.mark.parametrize("shape", TOO_SMALL, ids=shape_id)
@pytest.mark.parametrize(
    "name,dtype", DEBUG_BLOCK_MATCHING_BINDINGS, ids=binding_ids(DEBUG_BLOCK_MATCHING_BINDINGS)
)
def test_block_matching_binding_rejects_images_smaller_than_patch_size(name, dtype, shape):
    assert_too_small(lambda: block_matching_call(name, dtype, shape), shape)


@pytest.mark.parametrize("shape", AT_PATCH_SIZE, ids=shape_id)
@pytest.mark.parametrize(
    "name,dtype", DEBUG_BLOCK_MATCHING_BINDINGS, ids=binding_ids(DEBUG_BLOCK_MATCHING_BINDINGS)
)
def test_block_matching_binding_accepts_images_exactly_patch_size(name, dtype, shape):
    matches = block_matching_call(name, dtype, shape)
    # The reference patch matches itself at distance 0.
    assert (0, 0, 0.0) in matches
