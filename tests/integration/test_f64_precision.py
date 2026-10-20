"""
Integration tests for f64 (double precision) support in the Rust backend.
"""

import numpy as np
from numpy.testing import assert_allclose

from bm3dornl.bm3d_rust import (
    bm3d_hard_thresholding,
    bm3d_hard_thresholding_f64,
    bm3d_wiener_filtering_f64,
    bm3d_hard_thresholding_stack_f64,
    bm3d_wiener_filtering_stack_f64,
    estimate_streak_profile_rust,
    estimate_streak_profile_rust_f64,
)


class TestF64Import:
    def test_f64_functions_exist(self):
        assert callable(bm3d_hard_thresholding_f64)
        assert callable(bm3d_wiener_filtering_f64)
        assert callable(bm3d_hard_thresholding_stack_f64)
        assert callable(bm3d_wiener_filtering_stack_f64)
        assert callable(estimate_streak_profile_rust_f64)


class TestStreakProfileF64:
    def test_streak_profile_f64_dtype_preserved(self):
        image = np.random.rand(32, 64).astype(np.float64)
        result = estimate_streak_profile_rust_f64(image, 3.0, 3)

        assert result.dtype == np.float64
        assert result.shape == (64,)

    def test_streak_profile_f64_matches_f32_approximately(self):
        np.random.seed(42)
        image_f64 = np.random.rand(32, 64).astype(np.float64)
        image_f32 = image_f64.astype(np.float32)

        result_f64 = estimate_streak_profile_rust_f64(image_f64, 3.0, 3)
        result_f32 = estimate_streak_profile_rust(image_f32, 3.0, 3)

        assert_allclose(result_f64, result_f32.astype(np.float64), rtol=1e-4, atol=1e-6)

    def test_streak_profile_f64_uniform_image(self):
        image = np.full((32, 64), 0.5, dtype=np.float64)
        result = estimate_streak_profile_rust_f64(image, 3.0, 3)

        assert np.allclose(result, 0.0, atol=1e-10)


class TestBm3dHardThresholdingF64:
    def test_hard_thresholding_f64_dtype_preserved(self):
        np.random.seed(42)
        H, W = 32, 32
        input_noisy = np.random.rand(H, W).astype(np.float64)
        input_pilot = input_noisy.copy()
        sigma_psd = np.ones((H, W), dtype=np.float64)
        sigma_map = np.full((H, W), 0.1, dtype=np.float64)

        result = bm3d_hard_thresholding_f64(
            input_noisy,
            input_pilot,
            sigma_psd,
            sigma_map,
            sigma_random=0.1,
            threshold=2.7,
            patch_size=8,
            step_size=4,
            search_window=16,
            max_matches=16,
        )

        assert result.dtype == np.float64
        assert result.shape == (H, W)
        assert not np.isnan(result).any()

    def test_hard_thresholding_f64_vs_f32_similar(self):
        np.random.seed(42)
        H, W = 32, 32
        input_noisy_f64 = np.random.rand(H, W).astype(np.float64)
        input_noisy_f32 = input_noisy_f64.astype(np.float32)
        sigma_psd_f64 = np.ones((H, W), dtype=np.float64)
        sigma_psd_f32 = np.ones((H, W), dtype=np.float32)
        sigma_map_f64 = np.full((H, W), 0.1, dtype=np.float64)
        sigma_map_f32 = np.full((H, W), 0.1, dtype=np.float32)

        result_f64 = bm3d_hard_thresholding_f64(
            input_noisy_f64,
            input_noisy_f64,
            sigma_psd_f64,
            sigma_map_f64,
            sigma_random=0.1,
            threshold=2.7,
            patch_size=8,
            step_size=4,
            search_window=16,
            max_matches=16,
        )
        result_f32 = bm3d_hard_thresholding(
            input_noisy_f32,
            input_noisy_f32,
            sigma_psd_f32,
            sigma_map_f32,
            sigma_random=0.1,
            threshold=2.7,
            patch_size=8,
            step_size=4,
            search_window=16,
            max_matches=16,
        )

        assert_allclose(result_f64, result_f32.astype(np.float64), rtol=1e-3, atol=1e-4)


class TestBm3dWienerFilteringF64:
    def test_wiener_filtering_f64_dtype_preserved(self):
        np.random.seed(42)
        H, W = 32, 32
        input_noisy = np.random.rand(H, W).astype(np.float64)
        input_pilot = input_noisy.copy()
        sigma_psd = np.ones((H, W), dtype=np.float64)
        sigma_map = np.full((H, W), 0.1, dtype=np.float64)

        result = bm3d_wiener_filtering_f64(
            input_noisy,
            input_pilot,
            sigma_psd,
            sigma_map,
            sigma_random=0.1,
            patch_size=8,
            step_size=4,
            search_window=16,
            max_matches=16,
        )

        assert result.dtype == np.float64
        assert result.shape == (H, W)
        assert not np.isnan(result).any()


class TestBm3dStackF64:
    def test_hard_thresholding_stack_f64_dtype_preserved(self):
        np.random.seed(42)
        N, H, W = 4, 32, 32
        input_noisy = np.random.rand(N, H, W).astype(np.float64)
        input_pilot = input_noisy.copy()
        sigma_psd = np.ones((H, W), dtype=np.float64)
        sigma_map = np.full((N, H, W), 0.1, dtype=np.float64)

        result = bm3d_hard_thresholding_stack_f64(
            input_noisy,
            input_pilot,
            sigma_psd,
            sigma_map,
            sigma_random=0.1,
            threshold=2.7,
            patch_size=8,
            step_size=4,
            search_window=16,
            max_matches=16,
        )

        assert result.dtype == np.float64
        assert result.shape == (N, H, W)
        assert not np.isnan(result).any()

    def test_wiener_filtering_stack_f64_dtype_preserved(self):
        np.random.seed(42)
        N, H, W = 4, 32, 32
        input_noisy = np.random.rand(N, H, W).astype(np.float64)
        input_pilot = input_noisy.copy()
        sigma_psd = np.ones((H, W), dtype=np.float64)
        sigma_map = np.full((N, H, W), 0.1, dtype=np.float64)

        result = bm3d_wiener_filtering_stack_f64(
            input_noisy,
            input_pilot,
            sigma_psd,
            sigma_map,
            sigma_random=0.1,
            patch_size=8,
            step_size=4,
            search_window=16,
            max_matches=16,
        )

        assert result.dtype == np.float64
        assert result.shape == (N, H, W)
        assert not np.isnan(result).any()


class TestF64Precision:
    def test_f64_preserves_small_values(self):
        image = np.full((32, 32), 1e-10, dtype=np.float64)
        image[:, 16] = 1e-9

        result = estimate_streak_profile_rust_f64(image, 3.0, 3)

        assert result.dtype == np.float64
        assert np.max(np.abs(result)) < 1e-8

    def test_f64_roundtrip_precision(self):
        np.random.seed(42)
        image = np.random.rand(32, 32).astype(np.float64) * 1e-7 + 0.5

        result = estimate_streak_profile_rust_f64(image, 3.0, 3)

        assert result.dtype == np.float64
