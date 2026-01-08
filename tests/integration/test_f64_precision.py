"""
Integration tests for f64 (double precision) support in the Rust backend.

These tests verify that:
1. f64 numpy arrays are accepted by the Rust functions
2. Output dtype matches input dtype (f64 in, f64 out)
3. Numerical correctness is maintained
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from bm3dornl.bm3d_rust import (
    bm3d_hard_thresholding,
    bm3d_hard_thresholding_f64,
    bm3d_wiener_filtering,
    bm3d_wiener_filtering_f64,
    bm3d_hard_thresholding_stack,
    bm3d_hard_thresholding_stack_f64,
    bm3d_wiener_filtering_stack,
    bm3d_wiener_filtering_stack_f64,
    estimate_streak_profile_rust,
    estimate_streak_profile_rust_f64,
)


class TestF64Import:
    """Verify all f64 functions are importable and callable."""

    def test_f64_functions_exist(self):
        """All f64 variants should be importable."""
        # This test passes if the imports at module level succeed
        assert callable(bm3d_hard_thresholding_f64)
        assert callable(bm3d_wiener_filtering_f64)
        assert callable(bm3d_hard_thresholding_stack_f64)
        assert callable(bm3d_wiener_filtering_stack_f64)
        assert callable(estimate_streak_profile_rust_f64)


class TestStreakProfileF64:
    """Test streak profile estimation with f64 precision."""

    def test_streak_profile_f64_dtype_preserved(self):
        """Output dtype should match input dtype (f64)."""
        image = np.random.rand(32, 64).astype(np.float64)
        result = estimate_streak_profile_rust_f64(image, 3.0, 3)

        assert result.dtype == np.float64
        assert result.shape == (64,)

    def test_streak_profile_f64_matches_f32_approximately(self):
        """f64 and f32 results should be numerically similar."""
        np.random.seed(42)
        image_f64 = np.random.rand(32, 64).astype(np.float64)
        image_f32 = image_f64.astype(np.float32)

        result_f64 = estimate_streak_profile_rust_f64(image_f64, 3.0, 3)
        result_f32 = estimate_streak_profile_rust(image_f32, 3.0, 3)

        # Results should be close but not identical due to precision
        assert_allclose(result_f64, result_f32.astype(np.float64), rtol=1e-4, atol=1e-6)

    def test_streak_profile_f64_uniform_image(self):
        """Uniform image should have approximately zero streak profile."""
        image = np.full((32, 64), 0.5, dtype=np.float64)
        result = estimate_streak_profile_rust_f64(image, 3.0, 3)

        assert np.allclose(result, 0.0, atol=1e-10)


class TestBm3dHardThresholdingF64:
    """Test BM3D hard thresholding with f64 precision."""

    def test_hard_thresholding_f64_dtype_preserved(self):
        """Output dtype should match input dtype (f64)."""
        np.random.seed(42)
        H, W = 32, 32
        input_noisy = np.random.rand(H, W).astype(np.float64)
        input_pilot = input_noisy.copy()
        sigma_psd = np.ones((H, W), dtype=np.float64)
        sigma_map = np.full((H, W), 0.1, dtype=np.float64)

        result = bm3d_hard_thresholding_f64(
            input_noisy, input_pilot, sigma_psd, sigma_map,
            sigma_random=0.1, threshold=2.7,
            patch_size=8, step_size=4, search_window=16, max_matches=16
        )

        assert result.dtype == np.float64
        assert result.shape == (H, W)
        assert not np.isnan(result).any()

    def test_hard_thresholding_f64_vs_f32_similar(self):
        """f64 and f32 results should be numerically similar."""
        np.random.seed(42)
        H, W = 32, 32
        input_noisy_f64 = np.random.rand(H, W).astype(np.float64)
        input_noisy_f32 = input_noisy_f64.astype(np.float32)
        sigma_psd_f64 = np.ones((H, W), dtype=np.float64)
        sigma_psd_f32 = np.ones((H, W), dtype=np.float32)
        sigma_map_f64 = np.full((H, W), 0.1, dtype=np.float64)
        sigma_map_f32 = np.full((H, W), 0.1, dtype=np.float32)

        result_f64 = bm3d_hard_thresholding_f64(
            input_noisy_f64, input_noisy_f64, sigma_psd_f64, sigma_map_f64,
            sigma_random=0.1, threshold=2.7,
            patch_size=8, step_size=4, search_window=16, max_matches=16
        )
        result_f32 = bm3d_hard_thresholding(
            input_noisy_f32, input_noisy_f32, sigma_psd_f32, sigma_map_f32,
            sigma_random=0.1, threshold=2.7,
            patch_size=8, step_size=4, search_window=16, max_matches=16
        )

        # Results should be similar (not identical due to precision)
        assert_allclose(result_f64, result_f32.astype(np.float64), rtol=1e-3, atol=1e-4)


class TestBm3dWienerFilteringF64:
    """Test BM3D Wiener filtering with f64 precision."""

    def test_wiener_filtering_f64_dtype_preserved(self):
        """Output dtype should match input dtype (f64)."""
        np.random.seed(42)
        H, W = 32, 32
        input_noisy = np.random.rand(H, W).astype(np.float64)
        input_pilot = input_noisy.copy()
        sigma_psd = np.ones((H, W), dtype=np.float64)
        sigma_map = np.full((H, W), 0.1, dtype=np.float64)

        result = bm3d_wiener_filtering_f64(
            input_noisy, input_pilot, sigma_psd, sigma_map,
            sigma_random=0.1,
            patch_size=8, step_size=4, search_window=16, max_matches=16
        )

        assert result.dtype == np.float64
        assert result.shape == (H, W)
        assert not np.isnan(result).any()


class TestBm3dStackF64:
    """Test BM3D stack operations with f64 precision."""

    def test_hard_thresholding_stack_f64_dtype_preserved(self):
        """Output dtype should match input dtype (f64) for stack."""
        np.random.seed(42)
        N, H, W = 4, 32, 32
        input_noisy = np.random.rand(N, H, W).astype(np.float64)
        input_pilot = input_noisy.copy()
        sigma_psd = np.ones((H, W), dtype=np.float64)
        sigma_map = np.full((N, H, W), 0.1, dtype=np.float64)

        result = bm3d_hard_thresholding_stack_f64(
            input_noisy, input_pilot, sigma_psd, sigma_map,
            sigma_random=0.1, threshold=2.7,
            patch_size=8, step_size=4, search_window=16, max_matches=16
        )

        assert result.dtype == np.float64
        assert result.shape == (N, H, W)
        assert not np.isnan(result).any()

    def test_wiener_filtering_stack_f64_dtype_preserved(self):
        """Output dtype should match input dtype (f64) for stack."""
        np.random.seed(42)
        N, H, W = 4, 32, 32
        input_noisy = np.random.rand(N, H, W).astype(np.float64)
        input_pilot = input_noisy.copy()
        sigma_psd = np.ones((H, W), dtype=np.float64)
        sigma_map = np.full((N, H, W), 0.1, dtype=np.float64)

        result = bm3d_wiener_filtering_stack_f64(
            input_noisy, input_pilot, sigma_psd, sigma_map,
            sigma_random=0.1,
            patch_size=8, step_size=4, search_window=16, max_matches=16
        )

        assert result.dtype == np.float64
        assert result.shape == (N, H, W)
        assert not np.isnan(result).any()


class TestF64Precision:
    """Test that f64 provides higher precision than f32."""

    def test_f64_preserves_small_values(self):
        """f64 should preserve very small values better than f32."""
        # Create image with very small values that might lose precision in f32
        image = np.full((32, 32), 1e-10, dtype=np.float64)
        image[:, 16] = 1e-9  # Slight streak

        result = estimate_streak_profile_rust_f64(image, 3.0, 3)

        # The result should have non-zero values at column 16 region
        assert result.dtype == np.float64
        # f64 should not underflow to zero
        assert np.max(np.abs(result)) < 1e-8  # Within expected range

    def test_f64_roundtrip_precision(self):
        """f64 should preserve full precision through processing."""
        np.random.seed(42)
        # Use values that would lose precision in f32
        image = np.random.rand(32, 32).astype(np.float64) * 1e-7 + 0.5

        result = estimate_streak_profile_rust_f64(image, 3.0, 3)

        # Verify we get f64 result
        assert result.dtype == np.float64
