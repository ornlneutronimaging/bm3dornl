"""Guard tests for public BM3D default values."""

import inspect

import numpy as np

import bm3dornl.bm3d as bm3d_module

from bm3dornl.bm3d import bm3d_ring_artifact_removal


def test_bm3d_ring_artifact_removal_defaults_are_documented_defaults():
    signature = inspect.signature(bm3d_ring_artifact_removal)
    defaults = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }

    assert defaults["mode"] == "streak"
    assert defaults["sigma_random"] == 0.1
    assert defaults["patch_size"] == 8
    assert defaults["step_size"] == 4
    assert defaults["search_window"] == 24
    assert defaults["max_matches"] == 16
    assert defaults["threshold"] is None
    assert defaults["streak_sigma_smooth"] == 3.0
    assert defaults["streak_iterations"] == 2
    assert defaults["sigma_map_smoothing"] == 20.0
    assert defaults["streak_sigma_scale"] == 1.1
    assert defaults["psd_width"] == 0.6
    assert defaults["multiscale"] is False
    assert defaults["num_scales"] is None
    assert defaults["filter_strength"] == 1.0
    assert defaults["debin_iterations"] == 30


# Parameter order before search_step. Positional callers rely on it.
PUBLIC_PARAMETERS_BEFORE_SEARCH_STEP = [
    "sinogram",
    "mode",
    "sigma_random",
    "patch_size",
    "step_size",
    "search_window",
    "max_matches",
    "batch_size",
    "threshold",
    "streak_sigma_smooth",
    "streak_iterations",
    "sigma_map_smoothing",
    "streak_sigma_scale",
    "psd_width",
    "sigma_map",
    "multiscale",
    "num_scales",
    "filter_strength",
    "debin_iterations",
    "log_domain_input",
    "progress",
]


def test_search_step_default_is_none_and_keeps_existing_positions():
    parameters = inspect.signature(bm3d_ring_artifact_removal).parameters
    assert "search_step" in parameters
    assert parameters["search_step"].default is None
    names = list(parameters)
    assert (
        names[: len(PUBLIC_PARAMETERS_BEFORE_SEARCH_STEP)]
        == PUBLIC_PARAMETERS_BEFORE_SEARCH_STEP
    )


_KERNEL_2D = ["input_noisy", "input_pilot", "sigma_psd", "sigma_map", "sigma_random"]
_GEOMETRY = ["patch_size", "step_size", "search_window", "max_matches"]
_SINGLE_SCALE_2D = [
    "sinogram",
    "mode",
    "sigma_random",
    "patch_size",
    "step_size",
    "search_window",
    "max_matches",
    "threshold",
    "streak_sigma_smooth",
    "streak_iterations",
    "sigma_map_smoothing",
    "streak_sigma_scale",
    "psd_width",
    "fft_alpha",
    "notch_width",
]
_MULTISCALE_2D = [
    "sinogram",
    "num_scales",
    "filter_strength",
    "threshold",
    "debin_iterations",
    "patch_size",
    "step_size",
    "search_window",
    "max_matches",
    "sigma_random",
    "streak_sigma_smooth",
    "streak_iterations",
    "sigma_map_smoothing",
    "streak_sigma_scale",
    "psd_width",
    "log_domain_input",
]
# Binding parameter lists before search_step, f32 and f64 alike.
BINDING_PARAMETERS_BEFORE_SEARCH_STEP = {
    "bm3d_hard_thresholding": _KERNEL_2D + ["threshold"] + _GEOMETRY,
    "bm3d_wiener_filtering": _KERNEL_2D + _GEOMETRY,
    "bm3d_hard_thresholding_stack": _KERNEL_2D
    + ["threshold"]
    + _GEOMETRY
    + ["progress_callback"],
    "bm3d_wiener_filtering_stack": _KERNEL_2D + _GEOMETRY + ["progress_callback"],
    "bm3d_ring_artifact_removal_2d": _SINGLE_SCALE_2D,
    "multiscale_bm3d_streak_removal_2d": _MULTISCALE_2D,
}


def test_bindings_append_search_step_last_with_default_none():
    for base, before in BINDING_PARAMETERS_BEFORE_SEARCH_STEP.items():
        for name in (base, base + "_f64"):
            parameters = inspect.signature(
                getattr(bm3d_module.bm3d_rust, name)
            ).parameters
            assert list(parameters) == before + ["search_step"], name
            assert parameters["search_step"].default is None, name


def test_single_scale_threshold_default_delegates_to_rust_binding(monkeypatch):
    calls = {}

    def fake_single_scale(sinogram, mode, **kwargs):
        calls["threshold"] = kwargs["threshold"]
        return sinogram.copy()

    monkeypatch.setattr(
        bm3d_module.bm3d_rust,
        "bm3d_ring_artifact_removal_2d",
        fake_single_scale,
    )

    sinogram = np.zeros((8, 8), dtype=np.float32)
    bm3d_ring_artifact_removal(sinogram)

    assert calls["threshold"] is None


def test_multiscale_threshold_default_delegates_to_multiscale_binding(monkeypatch):
    calls = {}

    def fake_multiscale(sinogram, **kwargs):
        calls["threshold"] = kwargs["threshold"]
        return sinogram.copy()

    monkeypatch.setattr(
        bm3d_module.bm3d_rust,
        "multiscale_bm3d_streak_removal_2d",
        fake_multiscale,
    )

    sinogram = np.zeros((8, 8), dtype=np.float32)
    bm3d_ring_artifact_removal(sinogram, multiscale=True)

    assert calls["threshold"] is None


def test_stack_single_scale_threshold_default_resolves_to_single_scale_default(
    monkeypatch,
):
    calls = {}

    def fake_hard_thresholding_stack(
        input_noisy,
        input_pilot,
        sigma_psd,
        sigma_map,
        sigma_random,
        threshold,
        patch_size,
        step_size,
        search_window,
        max_matches,
        progress_callback=None,
        search_step=None,
    ):
        calls["threshold"] = threshold
        return input_noisy.copy()

    def fake_wiener_filtering_stack(
        input_noisy,
        input_pilot,
        sigma_psd,
        sigma_map,
        sigma_random,
        patch_size,
        step_size,
        search_window,
        max_matches,
        progress_callback=None,
        search_step=None,
    ):
        return input_noisy.copy()

    monkeypatch.setattr(
        bm3d_module.bm3d_rust,
        "bm3d_hard_thresholding_stack",
        fake_hard_thresholding_stack,
    )
    monkeypatch.setattr(
        bm3d_module.bm3d_rust,
        "bm3d_wiener_filtering_stack",
        fake_wiener_filtering_stack,
    )

    stack = np.zeros((2, 8, 8), dtype=np.float32)
    sigma_map = np.zeros_like(stack)
    bm3d_ring_artifact_removal(stack, mode="generic", sigma_map=sigma_map)

    assert calls["threshold"] == bm3d_module.DEFAULT_SINGLE_SCALE_THRESHOLD


def test_stack_multiscale_threshold_default_delegates_to_multiscale_binding(
    monkeypatch,
):
    calls = []

    def fake_multiscale(sinogram, **kwargs):
        calls.append(kwargs["threshold"])
        return sinogram.copy()

    monkeypatch.setattr(
        bm3d_module.bm3d_rust,
        "multiscale_bm3d_streak_removal_2d",
        fake_multiscale,
    )

    stack = np.zeros((2, 8, 8), dtype=np.float32)
    bm3d_ring_artifact_removal(stack, multiscale=True)

    assert calls == [None, None]
