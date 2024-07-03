import pytest
import numpy as np
from bm3dornl.bm3d import bm3d_ring_artifact_removal_ms


size_x = 256
size_y = 256


@pytest.fixture
def setup_sinogram():
    return np.random.rand(size_x, size_y)

@pytest.mark.cuda_required
def test_bm3d_ring_artifact_removal_ms(
    setup_sinogram,
):
    sinogram = setup_sinogram

    result = bm3d_ring_artifact_removal_ms(sinogram, k=4)

    assert result is not None

    r, c = result.shape

    assert c == size_x
    assert r == size_y

    result_single_pass = bm3d_ring_artifact_removal_ms(sinogram, k=0)
    assert result_single_pass is not None

    r, c = result_single_pass.shape

    assert c == size_x
    assert r == size_y
