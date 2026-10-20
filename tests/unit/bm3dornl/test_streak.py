#!/usr/bin/env python3

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from bm3dornl import bm3d_rust


def estimate_streak_profile_python(sinogram, sigma_smooth=3.0, iterations=3):
    z_clean = sinogram.copy()
    streak_acc = np.zeros(sinogram.shape[1], dtype=np.float32)

    for _ in range(iterations):
        z_smooth = gaussian_filter(z_clean, (sigma_smooth, 3.0))
        residual = z_clean - z_smooth
        streak_update = np.median(residual, axis=0)
        streak_update = gaussian_filter1d(streak_update, 1.0)
        streak_acc += streak_update
        correction = np.tile(streak_update, (sinogram.shape[0], 1))
        z_clean = z_clean - correction

    return streak_acc


class TestStreakProfileRustVsPython:
    def test_uniform_image(self):
        sinogram = np.full((32, 64), 0.5, dtype=np.float32)

        py_result = estimate_streak_profile_python(
            sinogram, sigma_smooth=3.0, iterations=3
        )
        rust_result = bm3d_rust.estimate_streak_profile_rust(sinogram, 3.0, 3)

        assert np.allclose(py_result, 0.0, atol=1e-5), (
            f"Python result not near zero: max={np.max(np.abs(py_result))}"
        )
        assert np.allclose(rust_result, 0.0, atol=1e-5), (
            f"Rust result not near zero: max={np.max(np.abs(rust_result))}"
        )

    def test_vertical_stripe_detection(self):
        sinogram = np.zeros((64, 128), dtype=np.float32)
        sinogram[:, 50] = 1.0

        py_result = estimate_streak_profile_python(
            sinogram, sigma_smooth=3.0, iterations=3
        )
        rust_result = bm3d_rust.estimate_streak_profile_rust(sinogram, 3.0, 3)

        py_peak_idx = np.argmax(py_result)
        rust_peak_idx = np.argmax(rust_result)

        assert abs(py_peak_idx - 50) <= 2, (
            f"Python peak at {py_peak_idx}, expected near 50"
        )
        assert abs(rust_peak_idx - 50) <= 2, (
            f"Rust peak at {rust_peak_idx}, expected near 50"
        )

    def test_multiple_stripes(self):
        sinogram = np.zeros((64, 128), dtype=np.float32)
        sinogram[:, 20] = 0.3
        sinogram[:, 60] = 1.0
        sinogram[:, 100] = 0.6

        py_result = estimate_streak_profile_python(
            sinogram, sigma_smooth=3.0, iterations=3
        )
        rust_result = bm3d_rust.estimate_streak_profile_rust(sinogram, 3.0, 3)

        assert py_result[60] > py_result[20], "Python: column 60 should be > column 20"
        assert py_result[60] > py_result[100], (
            "Python: column 60 should be > column 100"
        )
        assert rust_result[60] > rust_result[20], (
            "Rust: column 60 should be > column 20"
        )
        assert rust_result[60] > rust_result[100], (
            "Rust: column 60 should be > column 100"
        )

    def test_output_shape(self):
        for shape in [(32, 64), (64, 128), (100, 200)]:
            sinogram = np.random.rand(*shape).astype(np.float32)

            py_result = estimate_streak_profile_python(
                sinogram, sigma_smooth=3.0, iterations=2
            )
            rust_result = bm3d_rust.estimate_streak_profile_rust(sinogram, 3.0, 2)

            assert py_result.shape == (shape[1],), (
                f"Python output shape mismatch for input {shape}"
            )
            assert rust_result.shape == (shape[1],), (
                f"Rust output shape mismatch for input {shape}"
            )

    def test_horizontal_structure_ignored(self):
        sinogram = np.zeros((64, 64), dtype=np.float32)
        sinogram[32, :] = 1.0

        py_result = estimate_streak_profile_python(
            sinogram, sigma_smooth=3.0, iterations=3
        )
        rust_result = bm3d_rust.estimate_streak_profile_rust(sinogram, 3.0, 3)

        py_std = np.std(py_result)
        rust_std = np.std(rust_result)

        assert py_std < 0.1, f"Python std too high: {py_std}"
        assert rust_std < 0.1, f"Rust std too high: {rust_std}"

    def test_different_iterations(self):
        sinogram = np.zeros((32, 64), dtype=np.float32)
        sinogram[:, 30] = 1.0

        for iterations in [1, 2, 3, 5]:
            py_result = estimate_streak_profile_python(
                sinogram, sigma_smooth=3.0, iterations=iterations
            )
            rust_result = bm3d_rust.estimate_streak_profile_rust(
                sinogram, 3.0, iterations
            )

            assert py_result[30] > 0, (
                f"Python failed to detect streak with {iterations} iterations"
            )
            assert rust_result[30] > 0, (
                f"Rust failed to detect streak with {iterations} iterations"
            )

    def test_different_sigma_smooth(self):
        sinogram = np.zeros((32, 64), dtype=np.float32)
        sinogram[:, 30] = 1.0

        for sigma in [1.0, 3.0, 5.0, 10.0]:
            py_result = estimate_streak_profile_python(
                sinogram, sigma_smooth=sigma, iterations=3
            )
            rust_result = bm3d_rust.estimate_streak_profile_rust(sinogram, sigma, 3)

            assert np.max(py_result) > 0, f"Python failed with sigma={sigma}"
            assert np.max(rust_result) > 0, f"Rust failed with sigma={sigma}"

    def test_numerical_similarity_simple_case(self):
        np.random.seed(42)
        sinogram = np.random.rand(32, 64).astype(np.float32) * 0.1
        sinogram[:, 32] += 0.5

        py_result = estimate_streak_profile_python(
            sinogram, sigma_smooth=3.0, iterations=3
        )
        rust_result = bm3d_rust.estimate_streak_profile_rust(sinogram, 3.0, 3)

        correlation = np.corrcoef(py_result, rust_result)[0, 1]
        assert correlation > 0.8, f"Results not well correlated: {correlation}"

        py_peak = np.argmax(py_result)
        rust_peak = np.argmax(rust_result)
        assert abs(py_peak - rust_peak) <= 2, (
            f"Peak location differs: Python={py_peak}, Rust={rust_peak}"
        )


class TestStreakProfileRustEdgeCases:
    def test_small_image(self):
        sinogram = np.random.rand(4, 8).astype(np.float32)
        result = bm3d_rust.estimate_streak_profile_rust(sinogram, 1.0, 1)
        assert result.shape == (8,)

    def test_narrow_image(self):
        sinogram = np.random.rand(100, 10).astype(np.float32)
        result = bm3d_rust.estimate_streak_profile_rust(sinogram, 2.0, 2)
        assert result.shape == (10,)

    def test_wide_image(self):
        sinogram = np.random.rand(10, 200).astype(np.float32)
        result = bm3d_rust.estimate_streak_profile_rust(sinogram, 2.0, 2)
        assert result.shape == (200,)

    def test_single_row(self):
        sinogram = np.random.rand(1, 32).astype(np.float32)
        result = bm3d_rust.estimate_streak_profile_rust(sinogram, 1.0, 1)
        assert result.shape == (32,)

    def test_zero_sigma(self):
        sinogram = np.random.rand(16, 16).astype(np.float32)
        result = bm3d_rust.estimate_streak_profile_rust(sinogram, 0.0, 2)
        assert result.shape == (16,)

    def test_contiguous_array(self):
        sinogram = np.random.rand(64, 128).astype(np.float32)
        sinogram_view = sinogram[::2, ::2]
        assert not sinogram_view.flags["C_CONTIGUOUS"]

        sinogram_copy = np.ascontiguousarray(sinogram_view)
        result = bm3d_rust.estimate_streak_profile_rust(sinogram_copy, 2.0, 2)
        assert result.shape == (64,)
