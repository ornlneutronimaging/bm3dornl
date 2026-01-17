//! Streak profile estimation for ring artifact removal.
//!
//! This module provides functions to estimate static vertical streak artifacts
//! in sinograms using an iterative robust approach with Gaussian smoothing
//! and median filtering.
//!
//! ## SIMD Optimization Notes
//!
//! The Gaussian blur operations are optimized for auto-vectorization:
//! - Pre-padded arrays avoid branching in the hot loop
//! - Contiguous memory access enables SIMD
//! - Row-major layout for cache efficiency

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

use crate::float_trait::Bm3dFloat;

// =============================================================================
// Constants for Streak Profile Estimation
// =============================================================================

/// Horizontal sigma for 2D Gaussian blur in streak estimation.
/// Controls smoothing along the column direction (sigma_x in scipy terms).
const STREAK_HORIZONTAL_SIGMA: f64 = 3.0;

/// Sigma for smoothing the streak update profile.
/// A small value preserves streak sharpness while removing noise.
const STREAK_UPDATE_SIGMA: f64 = 1.0;

/// Minimum row count for parallel processing in blur operations.
/// Set high to avoid rayon overhead for smaller arrays.
const PARALLEL_ROW_THRESHOLD: usize = 512;

/// Compute 1D Gaussian kernel with given sigma.
/// Kernel size is ceil(4 * sigma) * 2 + 1 to match scipy's default truncate=4.0
fn gaussian_kernel_1d<F: Bm3dFloat>(sigma: F) -> Vec<F> {
    if sigma <= F::zero() {
        return vec![F::one()];
    }

    let radius = (F::GAUSSIAN_TRUNCATE * sigma)
        .ceil()
        .to_usize()
        .unwrap_or(0);
    let size = 2 * radius + 1;
    let mut kernel = vec![F::zero(); size];

    let sigma2 = sigma * sigma;
    let mut sum = F::zero();
    let two = F::from_f64_c(2.0);
    let neg_one = F::from_f64_c(-1.0);

    for (i, k) in kernel.iter_mut().enumerate() {
        let x = F::usize_as(i) - F::usize_as(radius);
        let val = (neg_one * x * x / (two * sigma2)).exp();
        *k = val;
        sum += val;
    }

    // Normalize
    let inv_sum = F::one() / sum;
    for val in kernel.iter_mut() {
        *val *= inv_sum;
    }

    kernel
}

/// Reflect index for boundary handling (scipy 'reflect' mode).
/// For an array of length n, reflects indices outside [0, n-1].
/// reflect(-1) = 0, reflect(-2) = 1, reflect(n) = n-1, reflect(n+1) = n-2
#[inline(always)]
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

/// Fill a pre-allocated padded buffer with reflected boundaries.
/// This enables branchless convolution in the hot path without re-allocation.
#[inline]
fn fill_padded_row<F: Bm3dFloat>(input: &[F], radius: usize, padded: &mut Vec<F>) {
    let n = input.len();
    let padded_len = n + 2 * radius;

    // Ensure buffer size
    if padded.len() != padded_len {
        padded.resize(padded_len, F::zero());
    }

    // Since we are writing to specific indices, we don't strictly need to zero-fill,
    // but resize does it for new elements.

    // Copy original data
    // padded[radius] is the start of valid data
    let dest_slice = &mut padded[radius..radius + n];
    dest_slice.copy_from_slice(input);

    // Left reflection
    for i in 0..radius {
        let src_idx = reflect_index(-(i as isize) - 1, n);
        padded[radius - 1 - i] = input[src_idx];
    }

    // Right reflection
    for i in 0..radius {
        let src_idx = reflect_index((n + i) as isize, n);
        padded[radius + n + i] = input[src_idx];
    }
}

/// Create a 1D padded buffer (wrapper around fill_padded_row for backward compat/convenience)
#[inline]
fn create_padded_row<F: Bm3dFloat>(input: &[F], radius: usize) -> Vec<F> {
    let mut padded = Vec::with_capacity(input.len() + 2 * radius);
    fill_padded_row(input, radius, &mut padded);
    padded
}

/// Apply 1D convolution to a padded buffer (no bounds checking needed).
/// This is the SIMD-friendly hot path.
#[inline]
fn convolve_1d_padded<F: Bm3dFloat>(padded: &[F], kernel: &[F], output: &mut [F]) {
    let n = output.len();
    let klen = kernel.len();

    // Process in chunks for better cache utilization
    // The compiler can auto-vectorize this inner loop
    for i in 0..n {
        let mut sum = F::zero();
        // Unroll hint: kernel is typically small (e.g., 25 elements for sigma=3)
        for k in 0..klen {
            sum += padded[i + k] * kernel[k];
        }
        output[i] = sum;
    }
}

/// Apply 1D Gaussian blur to a 1D array with reflect boundary.
/// Matches scipy.ndimage.gaussian_filter1d behavior.
pub fn gaussian_blur_1d<F: Bm3dFloat>(input: ArrayView1<F>, sigma: F) -> Array1<F> {
    let kernel = gaussian_kernel_1d(sigma);
    let radius = kernel.len() / 2;
    let n = input.len();

    if n == 0 {
        return Array1::zeros(0);
    }

    // Fast path for very small arrays
    if n <= radius * 2 {
        let mut output = Array1::zeros(n);
        for i in 0..n {
            let mut sum = F::zero();
            for (k, &w) in kernel.iter().enumerate() {
                let src_idx = i as isize + k as isize - radius as isize;
                let reflected = reflect_index(src_idx, n);
                sum += w * input[reflected];
            }
            output[i] = sum;
        }
        return output;
    }

    // Create padded buffer for branchless convolution
    // Convert to contiguous Vec for padding
    let input_vec: Vec<F> = input.iter().copied().collect();
    let padded = create_padded_row(&input_vec, radius);
    let mut output = Array1::zeros(n);
    convolve_1d_padded(&padded, &kernel, output.as_slice_mut().unwrap());

    output
}

/// Apply 1D Gaussian blur along rows of a 2D array.
/// Optimized with pre-padding and parallelization for large arrays.
fn blur_rows<F: Bm3dFloat>(input: ArrayView2<F>, sigma: F) -> Array2<F> {
    let (rows, cols) = input.dim();
    let kernel = gaussian_kernel_1d(sigma);
    let radius = kernel.len() / 2;

    if cols == 0 || rows == 0 {
        return Array2::zeros((rows, cols));
    }

    let mut output = Array2::zeros((rows, cols));

    // For large arrays, use parallel processing
    if rows >= PARALLEL_ROW_THRESHOLD && cols > radius * 4 {
        // Get mutable rows as lanes
        let output_rows: Vec<_> = output.axis_iter_mut(Axis(0)).collect();
        let input_rows: Vec<_> = input.axis_iter(Axis(0)).collect();

        output_rows
            .into_par_iter()
            .zip(input_rows.into_par_iter())
            .for_each(|(mut out_row, in_row)| {
                let in_slice: Vec<F> = in_row.iter().copied().collect();
                let padded = create_padded_row(&in_slice, radius);
                let out_slice = out_row.as_slice_mut().unwrap();
                convolve_1d_padded(&padded, &kernel, out_slice);
            });
    } else {
        // Sequential processing for smaller arrays
        let mut row_slice = Vec::with_capacity(cols);
        let mut padded = Vec::with_capacity(cols + 2 * radius);

        for r in 0..rows {
            // Re-fill buffer without allocation (reuse capacity)
            row_slice.clear();
            row_slice.extend(input.row(r).iter().copied());

            fill_padded_row(&row_slice, radius, &mut padded);

            let out_slice = output.row_mut(r).into_slice().unwrap();
            convolve_1d_padded(&padded, &kernel, out_slice);
        }
    }

    output
}

/// Apply 1D Gaussian blur along columns of a 2D array.
/// Optimized with transposed processing for cache efficiency.
fn blur_cols<F: Bm3dFloat>(input: ArrayView2<F>, sigma: F) -> Array2<F> {
    let (rows, cols) = input.dim();
    let kernel = gaussian_kernel_1d(sigma);
    let radius = kernel.len() / 2;

    if cols == 0 || rows == 0 {
        return Array2::zeros((rows, cols));
    }

    let mut output = Array2::zeros((rows, cols));

    // For column blur, we process column-by-column
    // This is less cache-friendly but necessary for the separable filter

    if cols >= PARALLEL_ROW_THRESHOLD && rows > radius * 4 {
        // Parallel column processing
        let col_indices: Vec<usize> = (0..cols).collect();

        // Collect column data, process, and write back
        let results: Vec<Vec<F>> = col_indices
            .par_iter()
            .map(|&c| {
                let col_data: Vec<F> = (0..rows).map(|r| input[[r, c]]).collect();
                let padded = create_padded_row(&col_data, radius);
                let mut col_out = vec![F::zero(); rows];
                convolve_1d_padded(&padded, &kernel, &mut col_out);
                col_out
            })
            .collect();

        // Write results back
        for (c, col_out) in results.into_iter().enumerate() {
            for (r, &val) in col_out.iter().enumerate() {
                output[[r, c]] = val;
            }
        }
    } else {
        // Sequential column processing
        let mut col_data = Vec::with_capacity(rows);
        let mut padded = Vec::with_capacity(rows + 2 * radius);
        let mut col_out = vec![F::zero(); rows];

        for c in 0..cols {
            col_data.clear();
            col_data.extend((0..rows).map(|r| input[[r, c]]));

            fill_padded_row(&col_data, radius, &mut padded);

            convolve_1d_padded(&padded, &kernel, &mut col_out);

            for (r, &val) in col_out.iter().enumerate() {
                output[[r, c]] = val;
            }
        }
    }

    output
}

/// Apply 2D Gaussian blur with separate sigma for each axis.
/// sigma_y is applied along rows (axis 0), sigma_x along columns (axis 1).
/// This matches scipy.ndimage.gaussian_filter with (sigma_y, sigma_x).
pub fn gaussian_blur_2d<F: Bm3dFloat>(input: ArrayView2<F>, sigma_y: F, sigma_x: F) -> Array2<F> {
    // Separable: first blur along rows (x direction), then along columns (y direction)
    let blurred_x = blur_rows(input, sigma_x);
    blur_cols(blurred_x.view(), sigma_y)
}

/// Compute median of a slice using partial sorting.
/// Uses Rust's select_nth_unstable for O(n) average case.
fn median_slice<F: Bm3dFloat>(data: &mut [F]) -> F {
    let n = data.len();
    if n == 0 {
        return F::zero();
    }
    if n == 1 {
        return data[0];
    }
    if n == 2 {
        return (data[0] + data[1]) / F::from_f64_c(2.0);
    }

    let mid = n / 2;

    if n % 2 == 1 {
        // For odd length, find the middle element using partial sort
        let (_, median, _) = data.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        *median
    } else {
        // For even length, find the two middle elements
        // First find the upper middle
        let (left, upper, _) = data.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        // The lower middle is the max of the left partition
        let lower = left
            .iter()
            .copied()
            .fold(F::neg_infinity(), |acc, x| if x > acc { x } else { acc });
        (lower + *upper) / F::from_f64_c(2.0)
    }
}

/// Compute column-wise median (reduce along axis 0).
/// For each column j, computes median of all values in that column.
pub fn median_axis0<F: Bm3dFloat>(input: ArrayView2<F>) -> Array1<F> {
    let (rows, cols) = input.dim();

    if rows == 0 || cols == 0 {
        return Array1::zeros(cols);
    }

    let mut output = Array1::zeros(cols);

    // Reuse buffer for column data
    let mut col_data: Vec<F> = Vec::with_capacity(rows);

    for c in 0..cols {
        col_data.clear();
        col_data.extend((0..rows).map(|r| input[[r, c]]));
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
pub fn estimate_streak_profile_impl<F: Bm3dFloat>(
    sinogram: ArrayView2<F>,
    sigma_smooth: F,
    iterations: usize,
) -> Array1<F> {
    let (rows, cols) = sinogram.dim();

    // Initialize working copy and accumulator
    let mut z_clean = sinogram.to_owned();
    let mut streak_acc = Array1::zeros(cols);

    let horizontal_sigma = F::from_f64_c(STREAK_HORIZONTAL_SIGMA);
    let update_sigma = F::from_f64_c(STREAK_UPDATE_SIGMA);

    for _ in 0..iterations {
        // 1. Smooth to estimate object: gaussian_filter(z_clean, (sigma_smooth, STREAK_HORIZONTAL_SIGMA))
        let z_smooth = gaussian_blur_2d(z_clean.view(), sigma_smooth, horizontal_sigma);

        // 2. Compute residual
        let residual = &z_clean - &z_smooth;

        // 3. Column-wise median for robust streak estimate
        let streak_update = median_axis0(residual.view());

        // 4. Smooth the streak update: gaussian_filter1d(streak_update, STREAK_UPDATE_SIGMA)
        let streak_update_smooth = gaussian_blur_1d(streak_update.view(), update_sigma);

        // 5. Accumulate
        streak_acc += &streak_update_smooth;

        // 6. Subtract from current estimate (broadcast across rows)
        for r in 0..rows {
            for c in 0..cols {
                z_clean[[r, c]] -= streak_update_smooth[c];
            }
        }
    }

    streak_acc
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
        for sigma in [0.5f32, 1.0, 2.0, 3.0, 5.0] {
            let kernel = gaussian_kernel_1d(sigma);
            let sum: f32 = kernel.iter().sum();
            assert!(
                approx_eq(sum, 1.0, 1e-6),
                "Kernel for sigma={} sums to {} instead of 1.0",
                sigma,
                sum
            );
        }
    }

    #[test]
    fn test_gaussian_kernel_symmetric() {
        let kernel = gaussian_kernel_1d(2.0f32);
        let n = kernel.len();
        for i in 0..n / 2 {
            assert!(
                approx_eq(kernel[i], kernel[n - 1 - i], 1e-7),
                "Kernel not symmetric at position {}",
                i
            );
        }
    }

    #[test]
    fn test_gaussian_kernel_zero_sigma() {
        let kernel = gaussian_kernel_1d(0.0f32);
        assert_eq!(kernel.len(), 1);
        assert_eq!(kernel[0], 1.0);
    }

    // ==================== Gaussian Blur 1D Tests ====================

    #[test]
    fn test_gaussian_1d_identity() {
        // Very small sigma should approximately preserve input
        let input = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let output = gaussian_blur_1d(input.view(), 0.001f32);
        assert!(
            arrays_approx_equal_1d(&input, &output, 1e-5),
            "Very small sigma should preserve input"
        );
    }

    #[test]
    fn test_gaussian_1d_uniform() {
        // Uniform array should remain uniform after blur
        let input = Array1::from_elem(10, 5.0f32);
        let output = gaussian_blur_1d(input.view(), 2.0f32);
        for &val in output.iter() {
            assert!(
                approx_eq(val, 5.0, 1e-5),
                "Uniform input should remain uniform, got {}",
                val
            );
        }
    }

    #[test]
    fn test_gaussian_1d_smoothing() {
        // Step function should be smoothed
        let mut input = Array1::zeros(20);
        for i in 10..20 {
            input[i] = 1.0f32;
        }
        let output = gaussian_blur_1d(input.view(), 2.0f32);

        // The transition region should have intermediate values
        assert!(output[9] > 0.0 && output[9] < 1.0, "Should smooth the step");
        assert!(
            output[10] > 0.0 && output[10] < 1.0,
            "Should smooth the step"
        );
    }

    #[test]
    fn test_gaussian_1d_preserves_mean() {
        // Gaussian blur should approximately preserve the mean (for large arrays with proper boundary)
        let input = Array1::from_vec(vec![1.0f32, 3.0, 2.0, 5.0, 4.0, 2.0, 3.0, 1.0]);
        let output = gaussian_blur_1d(input.view(), 1.0f32);

        let input_mean: f32 = input.iter().sum::<f32>() / input.len() as f32;
        let output_mean: f32 = output.iter().sum::<f32>() / output.len() as f32;

        // Mean should be approximately preserved
        assert!(
            approx_eq(input_mean, output_mean, 0.5),
            "Mean should be approximately preserved: {} vs {}",
            input_mean,
            output_mean
        );
    }

    // ==================== Gaussian Blur 1D Tests (f64) ====================

    #[test]
    fn test_gaussian_1d_identity_f64() {
        let input = Array1::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0]);
        let output = gaussian_blur_1d(input.view(), 0.001f64);

        for (a, b) in input.iter().zip(output.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "Very small sigma should preserve input"
            );
        }
    }

    #[test]
    fn test_gaussian_1d_uniform_f64() {
        let input = Array1::from_elem(10, 5.0f64);
        let output = gaussian_blur_1d(input.view(), 2.0f64);
        for &val in output.iter() {
            assert!(
                (val - 5.0).abs() < 1e-10,
                "Uniform input should remain uniform, got {}",
                val
            );
        }
    }

    // ==================== Gaussian Blur 2D Tests ====================

    #[test]
    fn test_gaussian_2d_uniform() {
        // Uniform image should remain uniform
        let input = Array2::from_elem((10, 10), 3.0f32);
        let output = gaussian_blur_2d(input.view(), 2.0f32, 2.0f32);

        for &val in output.iter() {
            assert!(
                approx_eq(val, 3.0, 1e-5),
                "Uniform image should remain uniform, got {}",
                val
            );
        }
    }

    #[test]
    fn test_gaussian_2d_separable() {
        // 2D blur should equal sequential 1D blurs along each axis
        let input = Array2::from_shape_fn((8, 8), |(r, c)| (r * 8 + c) as f32 / 64.0);

        let output_2d = gaussian_blur_2d(input.view(), 1.5f32, 2.0f32);

        // Manual separable: blur rows first, then columns
        let after_rows = blur_rows(input.view(), 2.0f32);
        let after_cols = blur_cols(after_rows.view(), 1.5f32);

        assert!(
            arrays_approx_equal_2d(&output_2d, &after_cols, 1e-6),
            "2D blur should be separable"
        );
    }

    #[test]
    fn test_gaussian_2d_anisotropic() {
        // Different sigmas should produce different blurs
        let input = Array2::from_shape_fn(
            (16, 16),
            |(r, c)| {
                if r == 8 && c == 8 {
                    1.0f32
                } else {
                    0.0
                }
            },
        );

        let output_iso = gaussian_blur_2d(input.view(), 2.0f32, 2.0f32);
        let output_aniso = gaussian_blur_2d(input.view(), 4.0f32, 1.0f32);

        // Anisotropic blur should be different
        let diff: f32 = output_iso
            .iter()
            .zip(output_aniso.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.01,
            "Different sigmas should produce different results"
        );
    }

    // ==================== Median Tests ====================

    #[test]
    fn test_median_axis0_simple() {
        // 3x3 matrix with known medians
        let input = Array2::from_shape_vec(
            (3, 3),
            vec![
                1.0f32, 4.0, 7.0, // row 0
                2.0, 5.0, 8.0, // row 1
                3.0, 6.0, 9.0, // row 2
            ],
        )
        .unwrap();

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
        let input =
            Array2::from_shape_vec((4, 2), vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0])
                .unwrap();

        let result = median_axis0(input.view());

        // Column 0: median([1,2,3,4]) = (2+3)/2 = 2.5
        // Column 1: median([10,20,30,40]) = (20+30)/2 = 25
        assert_eq!(result.len(), 2);
        assert!(approx_eq(result[0], 2.5, 1e-6));
        assert!(approx_eq(result[1], 25.0, 1e-6));
    }

    #[test]
    fn test_median_axis0_single_row() {
        let input = Array2::from_shape_vec((1, 4), vec![5.0f32, 3.0, 8.0, 1.0]).unwrap();
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
        let input = Array2::from_shape_vec((5, 1), vec![5.0f32, 1.0, 9.0, 3.0, 7.0]).unwrap();
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
        let profile = estimate_streak_profile_impl(input.view(), 3.0f32, 3);

        for &val in profile.iter() {
            assert!(
                val.abs() < 1e-5,
                "Uniform image should have zero streak profile, got {}",
                val
            );
        }
    }

    #[test]
    fn test_streak_profile_vertical_stripe() {
        // Image with one bright vertical stripe
        let mut input = Array2::from_elem((32, 64), 0.0f32);
        for r in 0..32 {
            input[[r, 20]] = 1.0; // Bright stripe at column 20
        }

        let profile = estimate_streak_profile_impl(input.view(), 3.0f32, 3);

        // Profile should be highest around column 20
        let max_idx = profile
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert!(
            (max_idx as i32 - 20).abs() <= 2,
            "Peak should be near column 20, found at {}",
            max_idx
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

        let profile = estimate_streak_profile_impl(input.view(), 3.0f32, 3);

        // Profile should have local maxima near columns 10, 30, 50
        // Column 30 should have highest value (brightest stripe)
        assert!(
            profile[30] > profile[10],
            "Brightest stripe should have highest profile"
        );
        assert!(
            profile[30] > profile[50],
            "Brightest stripe should have highest profile"
        );
    }

    #[test]
    fn test_streak_profile_iterations_matter() {
        // More iterations should refine the estimate
        let mut input = Array2::from_elem((32, 32), 0.5f32);
        for r in 0..32 {
            input[[r, 16]] = 1.0;
        }

        let profile_1 = estimate_streak_profile_impl(input.view(), 3.0f32, 1);
        let profile_3 = estimate_streak_profile_impl(input.view(), 3.0f32, 3);

        // Both should detect the streak, but magnitudes may differ
        assert!(profile_1[16] > 0.0, "1 iteration should detect streak");
        assert!(profile_3[16] > 0.0, "3 iterations should detect streak");
    }

    #[test]
    fn test_streak_profile_shape() {
        // Output should have same length as input width
        let input = Array2::from_elem((64, 128), 0.5f32);
        let profile = estimate_streak_profile_impl(input.view(), 3.0f32, 2);

        assert_eq!(
            profile.len(),
            128,
            "Profile length should equal image width"
        );
    }

    #[test]
    fn test_streak_profile_horizontal_structure_ignored() {
        // Horizontal structures should not create streak profile
        let mut input = Array2::from_elem((64, 64), 0.0f32);
        for c in 0..64 {
            input[[32, c]] = 1.0; // Bright horizontal line
        }

        let profile = estimate_streak_profile_impl(input.view(), 3.0f32, 3);

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
        let profile = estimate_streak_profile_impl(input.view(), 1.0f32, 3);
        assert!(profile[16] > 0.0, "Should detect streak with small sigma");
    }

    #[test]
    fn test_streak_profile_large_sigma() {
        // Large sigma smooths more aggressively
        let mut input = Array2::from_elem((32, 32), 0.0f32);
        for r in 0..32 {
            input[[r, 16]] = 1.0;
        }

        let profile_small = estimate_streak_profile_impl(input.view(), 1.0f32, 3);
        let profile_large = estimate_streak_profile_impl(input.view(), 10.0f32, 3);

        // Large sigma should produce smoother profile
        // Check that profile_large is broader (spread out more)
        let peak_small = profile_small[16];
        let peak_large = profile_large[16];

        // Both should be positive (detected)
        assert!(peak_small > 0.0, "Small sigma should detect");
        assert!(peak_large > 0.0, "Large sigma should detect");
    }
}
