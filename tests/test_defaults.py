"""Guard tests for public BM3D default values."""

import inspect

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
    assert defaults["threshold"] == 2.7
    assert defaults["streak_sigma_smooth"] == 3.0
    assert defaults["streak_iterations"] == 2
    assert defaults["sigma_map_smoothing"] == 20.0
    assert defaults["streak_sigma_scale"] == 1.1
    assert defaults["psd_width"] == 0.6
    assert defaults["multiscale"] is False
    assert defaults["num_scales"] is None
    assert defaults["filter_strength"] == 1.0
    assert defaults["debin_iterations"] == 30
