import pytest
import numpy as np
from bm3dornl import bm3d_ring_artifact_removal

@pytest.mark.parametrize("patch_size", [7, 8])
@pytest.mark.parametrize("is_stack", [True, False])
def test_rust_backend_execution(patch_size, is_stack):
    """
    Verify that the Rust backend runs correctly for different configurations.
    """
    H, W = 128, 128
    if is_stack:
        input_data = np.random.normal(0.5, 0.1, (16, H, W)).astype(np.float32)
    else:
        input_data = np.random.normal(0.5, 0.1, (H, W)).astype(np.float32)

    # Run BM3D with flat parameters (canonical names)
    output = bm3d_ring_artifact_removal(
        input_data,
        mode="streak",
        sigma_random=0.1,
        patch_size=patch_size,
        batch_size=4,  # Use smaller batch size to trigger batching logic
    )

    assert output.shape == input_data.shape
    assert output.dtype == np.float32
    assert not np.isnan(output).any()

    # Check that output is not identical to input (denoising happened)
    # Using small epsilon, but random noise should be smoothed.
    assert not np.allclose(output, input_data, atol=1e-6)

def test_rust_backend_chunking():
    """Verify chunking does not crash for stack larger than batch size."""
    H, W = 64, 64
    N = 10
    batch_size = 3

    input_stack = np.random.rand(N, H, W).astype(np.float32)
    output = bm3d_ring_artifact_removal(
        input_stack,
        mode="generic",
        sigma_random=0.1,
        patch_size=8,
        batch_size=batch_size,
    )
    assert output.shape == (N, H, W)
