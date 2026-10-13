"""The ``search_step`` keyword: per-mode default and validation (issue #163).

``step_size`` spaces the reference patches; ``search_step`` spaces the
candidate positions that block matching tests around each reference.
``search_step=None`` resolves per mode: every position (1) in generic mode
and ``step_size`` in streak, multiscale and Fourier-SVD mode. An explicit
value is used as given. The low-level kernel bindings do not know the mode,
so there ``None`` means ``step_size``. Every entry point rejects 0 with a
ValueError.
"""

import inspect

import numpy as np
import pytest

from bm3dornl import bm3d_ring_artifact_removal
from bm3dornl import bm3d_rust

SIGMA = 0.05


def noisy_step_edges(shape, seed=7, dtype=np.float32):
    """Step edges off the step-4 grid (column 41, row 35) plus Gaussian noise."""
    rows, cols = shape[-2:]
    truth = np.full((rows, cols), 0.25)
    truth[:, 41:] = 0.75
    truth[35:, :] += 0.15
    noise = np.random.default_rng(seed).normal(0.0, SIGMA, shape)
    return (truth + noise).astype(dtype)


def run(image, **kwargs):
    # bm3dornl applies sigma_random to the min/max-normalised image.
    sigma_norm = SIGMA / float(image.max() - image.min())
    return np.asarray(bm3d_ring_artifact_removal(image, sigma_random=sigma_norm, **kwargs))


SHAPES = {"2d": (64, 80), "3d": (2, 64, 80)}
# Multiscale needs width >= 80 for one binned scale; 160 gives two.
MULTISCALE_SHAPES = {"2d": (64, 160), "3d": (2, 64, 160)}


@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_generic_default_searches_every_position(dim):
    image = noisy_step_edges(SHAPES[dim])
    default = run(image, mode="generic")
    assert np.array_equal(default, run(image, mode="generic", search_step=1))
    assert not np.array_equal(default, run(image, mode="generic", search_step=4)), (
        "generic default still searches on the step_size grid"
    )


@pytest.mark.parametrize("step_size", [4, 2])
@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_streak_default_searches_on_step_size(dim, step_size):
    image = noisy_step_edges(SHAPES[dim])
    default = run(image, mode="streak", step_size=step_size)
    assert np.array_equal(
        default, run(image, mode="streak", step_size=step_size, search_step=step_size)
    )
    assert not np.array_equal(
        default, run(image, mode="streak", step_size=step_size, search_step=1)
    ), "streak default searches every position"


@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_multiscale_default_searches_on_step_size(dim):
    image = noisy_step_edges(MULTISCALE_SHAPES[dim])
    default = run(image, mode="streak", multiscale=True)
    assert np.array_equal(default, run(image, mode="streak", multiscale=True, search_step=4))
    assert not np.array_equal(
        default, run(image, mode="streak", multiscale=True, search_step=1)
    ), "multiscale default searches every position"


@pytest.mark.parametrize("dim", ["2d", "3d"])
@pytest.mark.parametrize("mode", ["generic", "streak"])
def test_explicit_search_step_is_used(mode, dim):
    image = noisy_step_edges(SHAPES[dim])
    explicit = run(image, mode=mode, search_step=2)
    for other in (None, 1, 4):
        assert not np.array_equal(explicit, run(image, mode=mode, search_step=other)), (
            f"{mode}: search_step=2 gave the same output as search_step={other}"
        )


# --- 3D path: the stride each stack kernel gets ------------------------------
#
# The 3D path calls two stack kernels. If one call drops search_step, both
# runs of an output comparison above use step_size in that call, so their
# outputs stay equal. These tests read the stride of each call instead.

STACK_KERNELS = ("bm3d_hard_thresholding_stack", "bm3d_wiener_filtering_stack")


def record_stack_search_steps(monkeypatch):
    """Wrap both stack kernels and record the search stride of each call.

    The wrappers call through to the real kernels. A kernel binding uses
    ``step_size`` when ``search_step`` is ``None`` or not passed.
    """
    seen = {name: [] for name in STACK_KERNELS}
    for name in STACK_KERNELS:
        real = getattr(bm3d_rust, name)

        def spy(*args, _real=real, _name=name, **kwargs):
            bound = inspect.signature(_real).bind(*args, **kwargs).arguments
            search_step = bound.get("search_step")
            seen[_name].append(bound["step_size"] if search_step is None else search_step)
            return _real(*args, **kwargs)

        monkeypatch.setattr(bm3d_rust, name, spy)
    return seen


# mode -> [(keyword arguments, stride both stack kernels must use)]
STACK_SEARCH_STEPS = {
    "generic": [({}, 1), ({"step_size": 2}, 1), ({"search_step": 4}, 4)],
    "streak": [({}, 4), ({"step_size": 2}, 2), ({"search_step": 1}, 1)],
    "fourier_svd": [({}, 4), ({"search_step": 2}, 2)],
}


@pytest.mark.parametrize("mode", STACK_SEARCH_STEPS)
def test_stack_path_gives_both_kernels_the_resolved_search_step(monkeypatch, mode):
    seen = record_stack_search_steps(monkeypatch)
    image = noisy_step_edges(SHAPES["3d"])
    for kwargs, expected in STACK_SEARCH_STEPS[mode]:
        for calls in seen.values():
            calls.clear()
        run(image, mode=mode, **kwargs)
        assert seen == {name: [expected] for name in STACK_KERNELS}, (
            f"3D {mode} with {kwargs}: the stack kernels got search strides {seen}, "
            f"expected {expected} in both"
        )


ENTRY_POINTS = {
    "2d-generic": ("2d", {"mode": "generic"}),
    "2d-streak": ("2d", {"mode": "streak"}),
    "2d-multiscale": ("2d", {"mode": "streak", "multiscale": True}),
    "3d-generic": ("3d", {"mode": "generic"}),
    "3d-streak": ("3d", {"mode": "streak"}),
    "3d-multiscale": ("3d", {"mode": "streak", "multiscale": True}),
}


def entry_point_image(dim, kwargs):
    shapes = MULTISCALE_SHAPES if kwargs.get("multiscale") else SHAPES
    return noisy_step_edges(shapes[dim])


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_zero_search_step_raises_value_error(entry):
    dim, kwargs = ENTRY_POINTS[entry]
    image = entry_point_image(dim, kwargs)
    # The keyword itself must be accepted; only the value 0 is invalid.
    run(image, search_step=1, **kwargs)
    with pytest.raises(ValueError, match="search_step"):
        run(image, search_step=0, **kwargs)


@pytest.mark.parametrize("bad", [-1, 2.5, "1"], ids=["negative", "float", "str"])
@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_non_positive_integer_search_step_raises(entry, bad):
    dim, kwargs = ENTRY_POINTS[entry]
    image = entry_point_image(dim, kwargs)
    run(image, search_step=1, **kwargs)
    with pytest.raises((TypeError, ValueError, OverflowError)):
        run(image, search_step=bad, **kwargs)


# --- Bindings ---------------------------------------------------------------


def kernel_call(name, dtype, step_size, **kwargs):
    """Call a low-level kernel binding with positional arguments as before #163."""
    fn = getattr(bm3d_rust, name)
    stack = "stack" in name
    image = noisy_step_edges((2, 64, 80) if stack else (64, 80), dtype=dtype)
    sigma_psd = np.zeros((1, 1), dtype=dtype)
    sigma_map = np.zeros((1, 1, 1) if stack else (1, 1), dtype=dtype)
    args = [image, image, sigma_psd, sigma_map, SIGMA]
    if "hard_thresholding" in name:
        args.append(2.7)
    args += [8, step_size, 24, 16]
    return np.asarray(fn(*args, **kwargs))


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


@pytest.mark.parametrize("name,dtype", KERNEL_BINDINGS, ids=[b[0] for b in KERNEL_BINDINGS])
def test_kernel_binding_search_step_none_means_step_size(name, dtype):
    # step_size 2 shows that None follows step_size rather than a fixed 4.
    positional = kernel_call(name, dtype, 2)
    assert np.array_equal(positional, kernel_call(name, dtype, 2, search_step=None))
    assert np.array_equal(positional, kernel_call(name, dtype, 2, search_step=2))
    assert not np.array_equal(positional, kernel_call(name, dtype, 2, search_step=1))


@pytest.mark.parametrize("name,dtype", KERNEL_BINDINGS, ids=[b[0] for b in KERNEL_BINDINGS])
def test_kernel_binding_rejects_zero_search_step(name, dtype):
    # A stride of 0 must not reach the search loop, where it would panic.
    with pytest.raises(ValueError, match="search_step"):
        kernel_call(name, dtype, 4, search_step=0)


HIGH_LEVEL_BINDINGS = [
    ("bm3d_ring_artifact_removal_2d", np.float32),
    ("bm3d_ring_artifact_removal_2d_f64", np.float64),
]


@pytest.mark.parametrize(
    "name,dtype", HIGH_LEVEL_BINDINGS, ids=[b[0] for b in HIGH_LEVEL_BINDINGS]
)
def test_high_level_binding_resolves_search_step_per_mode(name, dtype):
    fn = getattr(bm3d_rust, name)
    image = noisy_step_edges((64, 80), dtype=dtype)
    sigma_norm = SIGMA / float(image.max() - image.min())
    # mode -> (stride the default must equal, a stride it must differ from)
    for mode, (resolved, other) in {
        "generic": (1, 4),
        "streak": (4, 1),
        "fourier_svd": (4, 1),
    }.items():
        default = np.asarray(fn(image, mode, sigma_random=sigma_norm))
        assert np.array_equal(
            default, fn(image, mode, sigma_random=sigma_norm, search_step=resolved)
        ), f"{mode}: default differs from search_step={resolved}"
        assert not np.array_equal(
            default, fn(image, mode, sigma_random=sigma_norm, search_step=other)
        ), f"{mode}: default equals search_step={other}"


MULTISCALE_BINDINGS = [
    ("multiscale_bm3d_streak_removal_2d", np.float32),
    ("multiscale_bm3d_streak_removal_2d_f64", np.float64),
]


@pytest.mark.parametrize(
    "name,dtype", MULTISCALE_BINDINGS, ids=[b[0] for b in MULTISCALE_BINDINGS]
)
def test_multiscale_binding_default_search_step_is_step_size(name, dtype):
    fn = getattr(bm3d_rust, name)
    image = noisy_step_edges((64, 160), dtype=dtype)
    default = np.asarray(fn(image))
    assert np.array_equal(default, fn(image, search_step=4))
    assert not np.array_equal(default, fn(image, search_step=1))
