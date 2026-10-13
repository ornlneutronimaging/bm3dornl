"""Input arrays in any memory layout give the C-order result (issue #164).

The Rust fast paths read pixel data as a row-major slice. They took that
slice with ``as_slice_memory_order()``, which also accepts a Fortran-order or
negative-stride array, so such an input was read with rows and columns, or
the column order, mixed up (64 x 80 step image: RMSE 0.352 against the clean
image instead of 0.006). The public 2D path was safe because it calls
``np.ascontiguousarray`` first; the 3D path and the bindings were not.

Every entry point must return, for the same values in any layout, exactly
the output of a C-contiguous input: the bindings convert the input to
standard layout, so the same Rust code runs on the same data. The outputs
are compared bit for bit, within one process, so at the same thread count.

Layouts: Fortran order, the transpose of a C array, a negative-stride view
(``arr[..., ::-1]`` of a C array) and every second column of a wider array.
For stacks also a one-slice Fortran-order stack (the reported case) and a
two-slice stack whose slices are F-contiguous.
"""

import numpy as np
import pytest

from bm3dornl import bm3d_ring_artifact_removal, bm3d_rust
from bm3dornl.fourier_svd import fourier_svd_removal
from bm3dornl.utils import estimate_noise_sigma

SIGMA = 0.05
ROWS, COLS = 48, 64


def noisy_step_edges(shape, dtype=np.float32, seed=7):
    """Positive step edges off the patch grid, a ramp and Gaussian noise.

    Not square, and not symmetric, so a misread layout moves the edges.
    """
    rows, cols = shape[-2:]
    truth = np.full((rows, cols), 0.3) + 0.002 * np.arange(cols)
    truth[:, 29:] += 0.45
    truth[19:, :] += 0.1
    noise = np.random.default_rng(seed).normal(0.0, SIGMA, shape)
    return (truth + noise).astype(dtype)


def fortran(x):
    out = np.asfortranarray(x)
    assert out.flags.f_contiguous and not out.flags.c_contiguous
    return out


def transposed(x):
    """The transpose (last two axes) of a C array that holds the transposed values."""
    out = np.ascontiguousarray(np.swapaxes(x, -1, -2)).swapaxes(-1, -2)
    assert out.base is not None
    assert out[(0,) * (x.ndim - 2)].flags.f_contiguous
    assert not out[(0,) * (x.ndim - 2)].flags.c_contiguous
    return out


def reversed_columns(x):
    out = np.ascontiguousarray(x[..., ::-1])[..., ::-1]
    assert out.strides[-1] < 0
    return out


def every_second_column(x):
    # The skipped columns hold NaN, so reading them shows in the output.
    wide = np.full((*x.shape[:-1], 2 * x.shape[-1]), np.nan, dtype=x.dtype)
    wide[..., ::2] = x
    out = wide[..., ::2]
    assert not out.flags.c_contiguous and not out.flags.f_contiguous
    return out


LAYOUTS = {
    "fortran": fortran,
    "transposed": transposed,
    "reversed-columns": reversed_columns,
    "every-second-column": every_second_column,
}


def assert_same_output(run, values, layout):
    """``run`` on ``values`` in ``layout`` equals ``run`` on C-order values."""
    c_input = np.ascontiguousarray(values)
    expected = np.asarray(run(c_input))
    other = LAYOUTS[layout](values)
    assert np.array_equal(other, c_input)
    got = np.asarray(run(other))
    assert got.dtype == expected.dtype
    assert got.shape == expected.shape
    assert np.array_equal(got, expected), (
        f"{layout}: max |difference| {float(np.nanmax(np.abs(got - expected))):.3g}"
    )


def public_api(**kwargs):
    def run(x):
        return bm3d_ring_artifact_removal(x, sigma_random=SIGMA, **kwargs)

    return run


# --- Public API ----------------------------------------------------------------

# name -> (input shape, keyword arguments). Multiscale width 160 gives two
# binned scales.
PUBLIC = {
    "2d-generic": ((ROWS, COLS), {"mode": "generic"}),
    "2d-streak": ((ROWS, COLS), {"mode": "streak"}),
    "2d-fourier_svd": ((ROWS, COLS), {"mode": "fourier_svd"}),
    "2d-multiscale": ((ROWS, 160), {"mode": "streak", "multiscale": True}),
    "3d-generic": ((2, ROWS, COLS), {"mode": "generic"}),
    "3d-streak": ((2, ROWS, COLS), {"mode": "streak"}),
    "3d-multiscale": ((2, ROWS, 160), {"mode": "streak", "multiscale": True}),
}


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("entry", PUBLIC)
def test_public_api_output_does_not_depend_on_layout(entry, layout):
    shape, kwargs = PUBLIC[entry]
    assert_same_output(public_api(**kwargs), noisy_step_edges(shape), layout)


@pytest.mark.parametrize("mode", ["generic", "streak"])
def test_one_slice_fortran_order_stack(mode):
    # The reported case: np.asfortranarray of a one-slice stack.
    assert_same_output(public_api(mode=mode), noisy_step_edges((1, ROWS, COLS)), "fortran")


@pytest.mark.parametrize("mode", ["generic", "streak"])
def test_two_slice_stack_with_f_contiguous_slices(mode):
    # transpose(0, 2, 1) of a C array: each slice is F-contiguous.
    assert_same_output(
        public_api(mode=mode), noisy_step_edges((2, ROWS, COLS)), "transposed"
    )


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


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("mode", ["generic", "streak", "fourier_svd"])
@pytest.mark.parametrize(
    "name,dtype", HIGH_LEVEL_BINDINGS, ids=binding_ids(HIGH_LEVEL_BINDINGS)
)
def test_high_level_binding_output_does_not_depend_on_layout(name, dtype, mode, layout):
    fn = getattr(bm3d_rust, name)
    assert_same_output(
        lambda x: fn(x, mode, sigma_random=SIGMA),
        noisy_step_edges((ROWS, COLS), dtype=dtype),
        layout,
    )


@pytest.mark.parametrize("layout", LAYOUTS)
# Width 64 gives no binned scale (single-scale streak BM3D on the input
# itself); width 160 gives two.
@pytest.mark.parametrize("cols", [64, 160])
@pytest.mark.parametrize(
    "name,dtype", MULTISCALE_BINDINGS, ids=binding_ids(MULTISCALE_BINDINGS)
)
def test_multiscale_binding_output_does_not_depend_on_layout(name, dtype, cols, layout):
    fn = getattr(bm3d_rust, name)
    assert_same_output(fn, noisy_step_edges((ROWS, cols), dtype=dtype), layout)


STREAK_PROFILE_BINDINGS = [
    ("estimate_streak_profile_rust", np.float32),
    ("estimate_streak_profile_rust_f64", np.float64),
]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "name,dtype", STREAK_PROFILE_BINDINGS, ids=binding_ids(STREAK_PROFILE_BINDINGS)
)
def test_streak_profile_binding_output_does_not_depend_on_layout(name, dtype, layout):
    # sigma_smooth 3.0 and 3 iterations, the estimate_streak_profile defaults.
    fn = getattr(bm3d_rust, name)
    assert_same_output(
        lambda x: fn(x, 3.0, 3), noisy_step_edges((ROWS, COLS), dtype=dtype), layout
    )


def streak_psd(patch_size, dtype):
    """Anisotropic PSD of the form the 3D streak path builds (varies along rows)."""
    freq = np.arange(patch_size)
    profile = np.exp(-0.5 * (np.minimum(freq, patch_size - freq) / 0.6) ** 2)
    return np.tile(profile[:, None], (1, patch_size)).astype(dtype)


def kernel_runner(name, dtype, layout):
    """Run a kernel binding with every array argument in ``layout``.

    The PSD and the full noise map vary along one axis only, so a transposed
    read of either changes the output as well.
    """
    fn = getattr(bm3d_rust, name)
    stack = "stack" in name
    shape = (2, ROWS, COLS) if stack else (ROWS, COLS)
    noisy = noisy_step_edges(shape, dtype=dtype)
    # The Wiener pass takes a pilot estimate; another noise draw will do.
    pilot = noisy if "hard_thresholding" in name else noisy_step_edges(shape, dtype, seed=8)
    sigma_map = np.broadcast_to(0.01 + 0.002 * (np.arange(COLS) % 7), shape).astype(dtype)
    sigma_psd = streak_psd(8, dtype)

    def run(arrays):
        args = list(arrays) + [SIGMA]
        if "hard_thresholding" in name:
            args.append(2.7)
        args += [8, 3, 16, 16]
        return np.asarray(fn(*args))

    standard = [np.ascontiguousarray(a) for a in (noisy, pilot, sigma_psd, sigma_map)]
    return run, standard, [LAYOUTS[layout](a) for a in standard]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("name,dtype", KERNEL_BINDINGS, ids=binding_ids(KERNEL_BINDINGS))
def test_kernel_binding_output_does_not_depend_on_layout(name, dtype, layout):
    run, standard, other = kernel_runner(name, dtype, layout)
    for a, b in zip(standard, other):
        assert np.array_equal(a, b)
    expected = run(standard)
    got = run(other)
    assert got.dtype == expected.dtype == dtype
    assert np.array_equal(got, expected), (
        f"{layout}: max |difference| {float(np.nanmax(np.abs(got - expected))):.3g}"
    )


@pytest.mark.parametrize("name,dtype", KERNEL_BINDINGS, ids=binding_ids(KERNEL_BINDINGS))
def test_kernel_binding_one_slice_fortran_order(name, dtype):
    # The reported low-level case: a Fortran-order image (or one-slice stack).
    run, standard, _ = kernel_runner(name, dtype, "fortran")
    if "stack" in name:
        standard = [standard[0][:1], standard[1][:1], standard[2], standard[3][:1]]
        standard = [np.ascontiguousarray(a) for a in standard]
    expected = run(standard)
    got = run([np.asfortranarray(a) for a in standard])
    assert np.array_equal(got, expected), (
        f"max |difference| {float(np.nanmax(np.abs(got - expected))):.3g}"
    )


DEBUG_BLOCK_MATCHING_BINDINGS = [
    ("test_block_matching_rust", np.float32),
    ("test_block_matching_rust_f64", np.float64),
]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "name,dtype", DEBUG_BLOCK_MATCHING_BINDINGS, ids=binding_ids(DEBUG_BLOCK_MATCHING_BINDINGS)
)
def test_block_matching_binding_matches_do_not_depend_on_layout(name, dtype, layout):
    # Reference patch at (16, 24), patch 8, search window 16, 16 matches.
    fn = getattr(bm3d_rust, name)
    image = noisy_step_edges((ROWS, COLS), dtype=dtype)
    expected = fn(np.ascontiguousarray(image), 16, 24, 8, 16, 16)
    other = LAYOUTS[layout](image)
    assert np.array_equal(other, image)
    got = fn(other, 16, 24, 8, 16, 16)
    assert got == expected, f"{layout}: matches {got[:4]}... instead of {expected[:4]}..."


# --- Other array entry points --------------------------------------------------
#
# These wrappers pass the array to their binding as given, so they read the
# same Rust code paths (FFT, noise estimator) with the caller's layout.


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_noise_estimate_does_not_depend_on_layout(dtype, layout):
    image = noisy_step_edges((ROWS, COLS), dtype=dtype)
    expected = estimate_noise_sigma(np.ascontiguousarray(image))
    assert estimate_noise_sigma(LAYOUTS[layout](image)) == expected


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fourier_svd_removal_output_does_not_depend_on_layout(dtype, layout):
    assert_same_output(fourier_svd_removal, noisy_step_edges((ROWS, COLS), dtype=dtype), layout)
