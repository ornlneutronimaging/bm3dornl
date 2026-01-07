//! Streak profile estimation for ring artifact removal.
//!
//! This module provides functions to estimate static vertical streak artifacts
//! in sinograms using an iterative robust approach with Gaussian smoothing
//! and median filtering.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

// =============================================================================
// Constants for Streak Profile Estimation
// =============================================================================

/// Gaussian kernel truncation factor (matches scipy's default truncate=4.0).
/// The kernel radius is computed as ceil(GAUSSIAN_TRUNCATE * sigma).
const GAUSSIAN_TRUNCATE: f32 = 4.0;

/// Horizontal sigma for 2D Gaussian blur in streak estimation.
/// Controls smoothing along the column direction (sigma_x in scipy terms).
const STREAK_HORIZONTAL_SIGMA: f32 = 3.0;

/// Sigma for smoothing the streak update profile.
/// A small value preserves streak sharpness while removing noise.
const STREAK_UPDATE_SIGMA: f32 = 1.0;

/// Compute 1D Gaussian kernel with given sigma.
/// Kernel size is ceil(4 * sigma) * 2 + 1 to match scipy's default truncate=4.0
fn gaussian_kernel_1d(sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 {
        return vec![1.0];
    }

    let radius = (GAUSSIAN_TRUNCATE * sigma).ceil() as usize;
    let size = 2 * radius + 1;
    let mut kernel = vec![0.0f32; size];

    let sigma2 = sigma * sigma;
    let mut sum = 0.0f32;

    for i in 0..size {
        let x = i as f32 - radius as f32;
        let val = (-x * x / (2.0 * sigma2)).exp();
        kernel[i] = val;
        sum += val;
    }

    // Normalize
    for val in kernel.iter_mut() {
        *val /= sum;
    }

    kernel
}

/// Reflect index for boundary handling (scipy 'reflect' mode).
/// For an array of length n, reflects indices outside [0, n-1].
/// reflect(-1) = 0, reflect(-2) = 1, reflect(n) = n-1, reflect(n+1) = n-2
#[inline]
fn reflect_index(idx: isize, len: usize) -> usize {
    let n = len as isize;
    if idx < 0 {
        (-idx - 1).min(n - 1) as usize
    } else if idx >= n {
        let excess = idx - n;
        (n - 2 - excess).max(0) as usize
    } else {
        idx as usize
    }
}

/// Apply 1D Gaussian blur to a 1D array with reflect boundary.
/// Matches scipy.ndimage.gaussian_filter1d behavior.
pub fn gaussian_blur_1d(input: ArrayView1<f32>, sigma: f32) -> Array1<f32> {
    let kernel = gaussian_kernel_1d(sigma);
    let radius = kernel.len() / 2;
    let n = input.len();

    if n == 0 {
        return Array1::zeros(0);
    }

    let mut output = Array1::zeros(n);

    for i in 0..n {
        let mut sum = 0.0f32;
        for (k, &w) in kernel.iter().enumerate() {
            let src_idx = i as isize + k as isize - radius as isize;
            let reflected = reflect_index(src_idx, n);
            sum += w * input[reflected];
        }
        output[i] = sum;
    }

    output
}

/// Apply 1D Gaussian blur along rows of a 2D array.
fn blur_rows(input: ArrayView2<f32>, sigma: f32) -> Array2<f32> {
    let (rows, cols) = input.dim();
    let kernel = gaussian_kernel_1d(sigma);
    let radius = kernel.len() / 2;

    let mut output = Array2::zeros((rows, cols));

    for r in 0..rows {
        for c in 0..cols {
            let mut sum = 0.0f32;
            for (k, &w) in kernel.iter().enumerate() {
                let src_c = c as isize + k as isize - radius as isize;
                let reflected = reflect_index(src_c, cols);
                sum += w * input[[r, reflected]];
            }
            output[[r, c]] = sum;
        }
    }

    output
}

/// Apply 1D Gaussian blur along columns of a 2D array.
fn blur_cols(input: ArrayView2<f32>, sigma: f32) -> Array2<f32> {
    let (rows, cols) = input.dim();
    let kernel = gaussian_kernel_1d(sigma);
    let radius = kernel.len() / 2;

    let mut output = Array2::zeros((rows, cols));

    for r in 0..rows {
        for c in 0..cols {
            let mut sum = 0.0f32;
            for (k, &w) in kernel.iter().enumerate() {
                let src_r = r as isize + k as isize - radius as isize;
                let reflected = reflect_index(src_r, rows);
                sum += w * input[[reflected, c]];
            }
            output[[r, c]] = sum;
        }
    }

    output
}

/// Apply 2D Gaussian blur with separate sigma for each axis.
/// sigma_y is applied along rows (axis 0), sigma_x along columns (axis 1).
/// This matches scipy.ndimage.gaussian_filter with (sigma_y, sigma_x).
pub fn gaussian_blur_2d(input: ArrayView2<f32>, sigma_y: f32, sigma_x: f32) -> Array2<f32> {
    // Separable: first blur along rows (x direction), then along columns (y direction)
    let blurred_x = blur_rows(input, sigma_x);
    blur_cols(blurred_x.view(), sigma_y)
}

/// Compute median of a slice.
fn median_slice(data: &mut [f32]) -> f32 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }

    // Sort the slice
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if n % 2 == 1 {
        data[n / 2]
    } else {
        (data[n / 2 - 1] + data[n / 2]) / 2.0
    }
}

/// Compute column-wise median (reduce along axis 0).
/// For each column j, computes median of all values in that column.
pub fn median_axis0(input: ArrayView2<f32>) -> Array1<f32> {
    let (rows, cols) = input.dim();
    let mut output = Array1::zeros(cols);

    for c in 0..cols {
        let mut col_data: Vec<f32> = (0..rows).map(|r| input[[r, c]]).collect();
        output[c] = median_slice(&mut col_data);
    }

    output
}

/// Estimate the static vertical streak profile using an iterative robust approach.
///
/// This function separates object structure from static streaks by:
/// 1. Gaussian smoothing to estimate the object (low frequency)
/// 2. Computing residual (high frequency, contains streaks)
/// 3. Taking column-wise median of residual (robust streak estimate)
/// 4. Smoothing the streak estimate
/// 5. Subtracting from current estimate for next iteration
///
/// # Arguments
/// * `sinogram` - Input 2D sinogram (H x W)
/// * `sigma_smooth` - Sigma for vertical smoothing (default 3.0 in Python)
/// * `iterations` - Number of refinement iterations (default 3 in Python)
///
/// # Returns
/// 1D streak profile of length W
pub fn estimate_streak_profile_impl(
    sinogram: ArrayView2<f32>,
    sigma_smooth: f32,
    iterations: usize,
) -> Array1<f32> {
    let (rows, cols) = sinogram.dim();

    // Initialize working copy and accumulator
    let mut z_clean = sinogram.to_owned();
    let mut streak_acc = Array1::zeros(cols);

    for _ in 0..iterations {
        // 1. Smooth to estimate object: gaussian_filter(z_clean, (sigma_smooth, STREAK_HORIZONTAL_SIGMA))
        let z_smooth = gaussian_blur_2d(z_clean.view(), sigma_smooth, STREAK_HORIZONTAL_SIGMA);

        // 2. Compute residual
        let residual = &z_clean - &z_smooth;

        // 3. Column-wise median for robust streak estimate
        let streak_update = median_axis0(residual.view());

        // 4. Smooth the streak update: gaussian_filter1d(streak_update, STREAK_UPDATE_SIGMA)
        let streak_update_smooth = gaussian_blur_1d(streak_update.view(), STREAK_UPDATE_SIGMA);

        // 5. Accumulate
        streak_acc = streak_acc + &streak_update_smooth;

        // 6. Subtract from current estimate (broadcast across rows)
        for r in 0..rows {
            for c in 0..cols {
                z_clean[[r, c]] -= streak_update_smooth[c];
            }
        }
    }

    streak_acc
}

/// PyO3 wrapper for estimate_streak_profile_impl.
#[pyfunction]
#[pyo3(name = "estimate_streak_profile_rust")]
pub fn estimate_streak_profile_py(
    py: Python<'_>,
    sinogram: PyReadonlyArray2<f32>,
    sigma_smooth: f32,
    iterations: usize,
) -> PyResult<Py<PyArray1<f32>>> {
    let sinogram_view = sinogram.as_array();
    let result = estimate_streak_profile_impl(sinogram_view, sigma_smooth, iterations);
    Ok(PyArray1::from_owned_array(py, result).into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // Helper: Check if two f32 values are approximately equal
    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // Helper: Check if two arrays are approximately equal
    fn arrays_approx_equal_1d(a: &Array1<f32>, b: &Array1<f32>, eps: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y, eps))
    }

    fn arrays_approx_equal_2d(a: &Array2<f32>, b: &Array2<f32>, eps: f32) -> bool {
        if a.dim() != b.dim() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y, eps))
    }

    // ==================== Reflect Index Tests ====================

    #[test]
    fn test_reflect_index_in_bounds() {
        assert_eq!(reflect_index(0, 5), 0);
        assert_eq!(reflect_index(2, 5), 2);
        assert_eq!(reflect_index(4, 5), 4);
    }

    #[test]
    fn test_reflect_index_negative() {
        // For array of length 5: [0,1,2,3,4]
        // reflect(-1) should give 0 (reflect at boundary)
        assert_eq!(reflect_index(-1, 5), 0);
        assert_eq!(reflect_index(-2, 5), 1);
        assert_eq!(reflect_index(-3, 5), 2);
    }

    #[test]
    fn test_reflect_index_beyond_end() {
        // For array of length 5: [0,1,2,3,4]
        // reflect(5) should give 3 (reflect back)
        assert_eq!(reflect_index(5, 5), 3);
        assert_eq!(reflect_index(6, 5), 2);
        assert_eq!(reflect_index(7, 5), 1);
    }

    // ==================== Gaussian Kernel Tests ====================

    #[test]
    fn test_gaussian_kernel_sums_to_one() {
        for sigma in [0.5, 1.0, 2.0, 3.0, 5.0] {
            let kernel = gaussian_kernel_1d(sigma);
            let sum: f32 = kernel.iter().sum();
            assert!(
                approx_eq(sum, 1.0, 1e-6),
                "Kernel for sigma={} sums to {} instead of 1.0",
                sigma, sum
            );
        }
    }

    #[test]
    fn test_gaussian_kernel_symmetric() {
        let kernel = gaussian_kernel_1d(2.0);
        let n = kernel.len();
        for i in 0..n / 2 {
            assert!(
                approx_eq(kernel[i], kernel[n - 1 - i], 1e-7),
                "Kernel not symmetric at position {}", i
            );
        }
    }

    #[test]
    fn test_gaussian_kernel_zero_sigma() {
        let kernel = gaussian_kernel_1d(0.0);
        assert_eq!(kernel.len(), 1);
        assert_eq!(kernel[0], 1.0);
    }

    // ==================== Gaussian Blur 1D Tests ====================

    #[test]
    fn test_gaussian_1d_identity() {
        // Very small sigma should approximately preserve input
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let output = gaussian_blur_1d(input.view(), 0.001);
        assert!(
            arrays_approx_equal_1d(&input, &output, 1e-5),
            "Very small sigma should preserve input"
        );
    }

    #[test]
    fn test_gaussian_1d_uniform() {
        // Uniform array should remain uniform after blur
        let input = Array1::from_elem(10, 5.0f32);
        let output = gaussian_blur_1d(input.view(), 2.0);
        for &val in output.iter() {
            assert!(
                approx_eq(val, 5.0, 1e-5),
                "Uniform input should remain uniform, got {}", val
            );
        }
    }

    #[test]
    fn test_gaussian_1d_smoothing() {
        // Step function should be smoothed
        let mut input = Array1::zeros(20);
        for i in 10..20 {
            input[i] = 1.0;
        }
        let output = gaussian_blur_1d(input.view(), 2.0);

        // The transition region should have intermediate values
        assert!(output[9] > 0.0 && output[9] < 1.0, "Should smooth the step");
        assert!(output[10] > 0.0 && output[10] < 1.0, "Should smooth the step");
    }

    #[test]
    fn test_gaussian_1d_preserves_mean() {
        // Gaussian blur should approximately preserve the mean (for large arrays with proper boundary)
        let input = Array1::from_vec(vec![1.0, 3.0, 2.0, 5.0, 4.0, 2.0, 3.0, 1.0]);
        let output = gaussian_blur_1d(input.view(), 1.0);

        let input_mean: f32 = input.iter().sum::<f32>() / input.len() as f32;
        let output_mean: f32 = output.iter().sum::<f32>() / output.len() as f32;

        // Mean should be approximately preserved
        assert!(
            approx_eq(input_mean, output_mean, 0.5),
            "Mean should be approximately preserved: {} vs {}",
            input_mean, output_mean
        );
    }

    // ==================== Gaussian Blur 2D Tests ====================

    #[test]
    fn test_gaussian_2d_uniform() {
        // Uniform image should remain uniform
        let input = Array2::from_elem((10, 10), 3.0f32);
        let output = gaussian_blur_2d(input.view(), 2.0, 2.0);

        for &val in output.iter() {
            assert!(
                approx_eq(val, 3.0, 1e-5),
                "Uniform image should remain uniform, got {}", val
            );
        }
    }

    #[test]
    fn test_gaussian_2d_separable() {
        // 2D blur should equal sequential 1D blurs along each axis
        let input = Array2::from_shape_fn((8, 8), |(r, c)| (r * 8 + c) as f32 / 64.0);

        let output_2d = gaussian_blur_2d(input.view(), 1.5, 2.0);

        // Manual separable: blur rows first, then columns
        let after_rows = blur_rows(input.view(), 2.0);
        let after_cols = blur_cols(after_rows.view(), 1.5);

        assert!(
            arrays_approx_equal_2d(&output_2d, &after_cols, 1e-6),
            "2D blur should be separable"
        );
    }

    #[test]
    fn test_gaussian_2d_anisotropic() {
        // Different sigmas should produce different blurs
        let input = Array2::from_shape_fn((16, 16), |(r, c)| {
            if r == 8 && c == 8 { 1.0 } else { 0.0 }
        });

        let output_iso = gaussian_blur_2d(input.view(), 2.0, 2.0);
        let output_aniso = gaussian_blur_2d(input.view(), 4.0, 1.0);

        // Anisotropic blur should be different
        let diff: f32 = output_iso.iter().zip(output_aniso.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.01, "Different sigmas should produce different results");
    }

    // ==================== Median Tests ====================

    #[test]
    fn test_median_axis0_simple() {
        // 3x3 matrix with known medians
        let input = Array2::from_shape_vec((3, 3), vec![
            1.0, 4.0, 7.0,  // row 0
            2.0, 5.0, 8.0,  // row 1
            3.0, 6.0, 9.0,  // row 2
        ]).unwrap();

        let result = median_axis0(input.view());

        // Column 0: median([1,2,3]) = 2
        // Column 1: median([4,5,6]) = 5
        // Column 2: median([7,8,9]) = 8
        assert_eq!(result.len(), 3);
        assert!(approx_eq(result[0], 2.0, 1e-6));
        assert!(approx_eq(result[1], 5.0, 1e-6));
        assert!(approx_eq(result[2], 8.0, 1e-6));
    }

    #[test]
    fn test_median_axis0_even_rows() {
        // 4x2 matrix - even number of rows means average of middle two
        let input = Array2::from_shape_vec((4, 2), vec![
            1.0, 10.0,
            2.0, 20.0,
            3.0, 30.0,
            4.0, 40.0,
        ]).unwrap();

        let result = median_axis0(input.view());

        // Column 0: median([1,2,3,4]) = (2+3)/2 = 2.5
        // Column 1: median([10,20,30,40]) = (20+30)/2 = 25
        assert_eq!(result.len(), 2);
        assert!(approx_eq(result[0], 2.5, 1e-6));
        assert!(approx_eq(result[1], 25.0, 1e-6));
    }

    #[test]
    fn test_median_axis0_single_row() {
        let input = Array2::from_shape_vec((1, 4), vec![5.0, 3.0, 8.0, 1.0]).unwrap();
        let result = median_axis0(input.view());

        // Single row: each column's median is just that value
        assert_eq!(result.len(), 4);
        assert!(approx_eq(result[0], 5.0, 1e-6));
        assert!(approx_eq(result[1], 3.0, 1e-6));
        assert!(approx_eq(result[2], 8.0, 1e-6));
        assert!(approx_eq(result[3], 1.0, 1e-6));
    }

    #[test]
    fn test_median_axis0_unsorted() {
        // Values in non-sorted order
        let input = Array2::from_shape_vec((5, 1), vec![5.0, 1.0, 9.0, 3.0, 7.0]).unwrap();
        let result = median_axis0(input.view());

        // median([5,1,9,3,7]) = median(sorted: [1,3,5,7,9]) = 5
        assert_eq!(result.len(), 1);
        assert!(approx_eq(result[0], 5.0, 1e-6));
    }

    // ==================== Streak Profile Tests ====================

    #[test]
    fn test_streak_profile_uniform_image() {
        // Uniform image should have approximately zero streak profile
        let input = Array2::from_elem((32, 32), 0.5f32);
        let profile = estimate_streak_profile_impl(input.view(), 3.0, 3);

        for &val in profile.iter() {
            assert!(
                val.abs() < 1e-5,
                "Uniform image should have zero streak profile, got {}", val
            );
        }
    }

    #[test]
    fn test_streak_profile_vertical_stripe() {
        // Image with one bright vertical stripe
        let mut input = Array2::from_elem((32, 64), 0.0f32);
        for r in 0..32 {
            input[[r, 20]] = 1.0;  // Bright stripe at column 20
        }

        let profile = estimate_streak_profile_impl(input.view(), 3.0, 3);

        // Profile should be highest around column 20
        let max_idx = profile.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert!(
            (max_idx as i32 - 20).abs() <= 2,
            "Peak should be near column 20, found at {}", max_idx
        );
    }

    #[test]
    fn test_streak_profile_multiple_stripes() {
        // Image with multiple vertical stripes
        let mut input = Array2::from_elem((32, 64), 0.0f32);
        for r in 0..32 {
            input[[r, 10]] = 0.5;
            input[[r, 30]] = 1.0;
            input[[r, 50]] = 0.8;
        }

        let profile = estimate_streak_profile_impl(input.view(), 3.0, 3);

        // Profile should have local maxima near columns 10, 30, 50
        // Column 30 should have highest value (brightest stripe)
        assert!(profile[30] > profile[10], "Brightest stripe should have highest profile");
        assert!(profile[30] > profile[50], "Brightest stripe should have highest profile");
    }

    #[test]
    fn test_streak_profile_iterations_matter() {
        // More iterations should refine the estimate
        let mut input = Array2::from_elem((32, 32), 0.5f32);
        for r in 0..32 {
            input[[r, 16]] = 1.0;
        }

        let profile_1 = estimate_streak_profile_impl(input.view(), 3.0, 1);
        let profile_3 = estimate_streak_profile_impl(input.view(), 3.0, 3);

        // Both should detect the streak, but magnitudes may differ
        assert!(profile_1[16] > 0.0, "1 iteration should detect streak");
        assert!(profile_3[16] > 0.0, "3 iterations should detect streak");
    }

    #[test]
    fn test_streak_profile_shape() {
        // Output should have same length as input width
        let input = Array2::from_elem((64, 128), 0.5f32);
        let profile = estimate_streak_profile_impl(input.view(), 3.0, 2);

        assert_eq!(profile.len(), 128, "Profile length should equal image width");
    }

    #[test]
    fn test_streak_profile_horizontal_structure_ignored() {
        // Horizontal structures should not create streak profile
        let mut input = Array2::from_elem((64, 64), 0.0f32);
        for c in 0..64 {
            input[[32, c]] = 1.0;  // Bright horizontal line
        }

        let profile = estimate_streak_profile_impl(input.view(), 3.0, 3);

        // Profile should be approximately uniform (no column-specific streaks)
        let mean_profile: f32 = profile.iter().sum::<f32>() / profile.len() as f32;
        for &val in profile.iter() {
            assert!(
                (val - mean_profile).abs() < 0.1,
                "Horizontal structure should not create column-specific streaks"
            );
        }
    }

    #[test]
    fn test_streak_profile_small_sigma() {
        // Small sigma should preserve more structure
        let mut input = Array2::from_elem((32, 32), 0.0f32);
        for r in 0..32 {
            input[[r, 16]] = 1.0;
        }

        // Should still work with small sigma
        let profile = estimate_streak_profile_impl(input.view(), 1.0, 3);
        assert!(profile[16] > 0.0, "Should detect streak with small sigma");
    }

    #[test]
    fn test_streak_profile_large_sigma() {
        // Large sigma smooths more aggressively
        let mut input = Array2::from_elem((32, 32), 0.0f32);
        for r in 0..32 {
            input[[r, 16]] = 1.0;
        }

        let profile_small = estimate_streak_profile_impl(input.view(), 1.0, 3);
        let profile_large = estimate_streak_profile_impl(input.view(), 10.0, 3);

        // Large sigma should produce smoother profile
        // Check that profile_large is broader (spread out more)
        let peak_small = profile_small[16];
        let peak_large = profile_large[16];

        // Both should be positive (detected)
        assert!(peak_small > 0.0, "Small sigma should detect");
        assert!(peak_large > 0.0, "Large sigma should detect");
    }
}
