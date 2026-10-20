"""In generic mode, the default search at ``step_size=4`` keeps at most 10%
more bias than ``step_size=1``.
"""

import numpy as np
import pytest

from bm3dornl import bm3d_ring_artifact_removal

ROWS, COLS = 64, 80
# Neither edge sits on a multiple of the default step (4), so grid
# alignment cannot hide the search stride.
EDGE_COL, EDGE_ROW = 41, 35
DRAWS = 96
MAX_BIAS_RATIO = 1.10


def step_edge_truth() -> np.ndarray:
    truth = np.full((ROWS, COLS), 0.25, dtype=np.float64)
    truth[:, EDGE_COL:] = 0.75
    truth[EDGE_ROW:, :] += 0.15
    return truth


def filtered_draws(
    noisy: np.ndarray, sigma: float, step_size: int, dim: str
) -> np.ndarray:
    """Generic-mode output for each draw in ``noisy`` (shape (DRAWS, ROWS, COLS))."""
    if dim == "2d":
        outs = np.empty(noisy.shape, dtype=np.float64)
        for d, image in enumerate(noisy):
            # bm3dornl applies sigma_random to the min/max-normalised image.
            sigma_norm = sigma / float(image.max() - image.min())
            outs[d] = bm3d_ring_artifact_removal(
                image, mode="generic", sigma_random=sigma_norm, step_size=step_size
            )
        return outs
    # The 3D path normalises with the min/max of the whole stack.
    sigma_norm = sigma / float(noisy.max() - noisy.min())
    outs = bm3d_ring_artifact_removal(
        noisy, mode="generic", sigma_random=sigma_norm, step_size=step_size
    )
    return np.asarray(outs, dtype=np.float64)


def corrected_bias(sigma: float, step_size: int, dim: str = "2d") -> float:
    """Noise-corrected RMS bias of generic mode over DRAWS fixed-seed draws."""
    truth = step_edge_truth()
    noisy = np.stack(
        [
            (
                truth + np.random.default_rng(1000 + d).normal(0.0, sigma, truth.shape)
            ).astype(np.float32)
            for d in range(DRAWS)
        ]
    )
    outs = filtered_draws(noisy, sigma, step_size, dim)
    mean = outs.mean(axis=0)
    var = outs.var(axis=0, ddof=1)
    return float(np.sqrt(max(0.0, np.mean((mean - truth) ** 2) - np.mean(var) / DRAWS)))


@pytest.mark.parametrize("dim", ["2d", "3d"])
@pytest.mark.parametrize("sigma", [0.05, 0.1])
def test_default_search_bias_close_to_dense_reference_grid(sigma, dim):
    bias_step4 = corrected_bias(sigma, step_size=4, dim=dim)
    bias_step1 = corrected_bias(sigma, step_size=1, dim=dim)
    assert bias_step1 > 0.0, (
        f"{dim} sigma {sigma}: step_size=1 bias is below the noise floor"
    )
    ratio = bias_step4 / bias_step1
    print(
        f"{dim} sigma {sigma}: bias step 4 {bias_step4:.5f}, step 1 {bias_step1:.5f}, "
        f"ratio {ratio:.3f}"
    )
    assert ratio <= MAX_BIAS_RATIO, (
        f"{dim} sigma {sigma}: generic-mode bias at step_size=4 is {ratio:.3f}x the bias at "
        f"step_size=1 (bound {MAX_BIAS_RATIO}). A candidate search stepped by step_size "
        f"gives about 3x, and a stepped search in only one of the two BM3D stages gives "
        f"1.3x to 2.4x; a ratio just above the bound can also come from any other "
        f"change to grouping or aggregation at step_size=4, such as duplicated references"
    )
