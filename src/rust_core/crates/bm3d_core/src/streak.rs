//! Streak profile estimation for ring artifact removal.
//!
//! This module provides functions to estimate static vertical streak artifacts
//! in sinograms using an iterative approach with Gaussian smoothing.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

use crate::float_trait::Bm3dFloat;

const STREAK_HORIZONTAL_SIGMA: f64 = 3.0;

/// A small value preserves streak sharpness while removing noise.
const STREAK_UPDATE_SIGMA: f64 = 1.0;

const PARALLEL_ROW_THRESHOLD: usize = 512;

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

    let inv_sum = F::one() / sum;
    for val in kernel.iter_mut() {
        *val *= inv_sum;
    }

    kernel
}

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

#[inline]
fn fill_padded_row<F: Bm3dFloat>(input: &[F], radius: usize, padded: &mut Vec<F>) {
    let n = input.len();
    let padded_len = n + 2 * radius;

    if padded.len() != padded_len {
        padded.resize(padded_len, F::zero());
    }

    let dest_slice = &mut padded[radius..radius + n];
    dest_slice.copy_from_slice(input);

    for i in 0..radius {
        let src_idx = reflect_index(-(i as isize) - 1, n);
        padded[radius - 1 - i] = input[src_idx];
    }

    for i in 0..radius {
        let src_idx = reflect_index((n + i) as isize, n);
        padded[radius + n + i] = input[src_idx];
    }
}

#[inline]
fn create_padded_row<F: Bm3dFloat>(input: &[F], radius: usize) -> Vec<F> {
    let mut padded = Vec::with_capacity(input.len() + 2 * radius);
    fill_padded_row(input, radius, &mut padded);
    padded
}

#[inline]
fn convolve_1d_padded<F: Bm3dFloat>(padded: &[F], kernel: &[F], output: &mut [F]) {
    let n = output.len();
    let klen = kernel.len();

    for i in 0..n {
        let mut sum = F::zero();
        for k in 0..klen {
            sum += padded[i + k] * kernel[k];
        }
        output[i] = sum;
    }
}

pub fn gaussian_blur_1d<F: Bm3dFloat>(input: ArrayView1<F>, sigma: F) -> Array1<F> {
    let kernel = gaussian_kernel_1d(sigma);
    let radius = kernel.len() / 2;
    let n = input.len();

    if n == 0 {
        return Array1::zeros(0);
    }

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

    let input_vec: Vec<F> = input.iter().copied().collect();
    let padded = create_padded_row(&input_vec, radius);
    let mut output = Array1::zeros(n);
    convolve_1d_padded(&padded, &kernel, output.as_slice_mut().unwrap());

    output
}

fn blur_rows<F: Bm3dFloat>(input: ArrayView2<F>, sigma: F) -> Array2<F> {
    let (rows, cols) = input.dim();
    let kernel = gaussian_kernel_1d(sigma);
    let radius = kernel.len() / 2;

    if cols == 0 || rows == 0 {
        return Array2::zeros((rows, cols));
    }

    let mut output = Array2::zeros((rows, cols));

    if rows >= PARALLEL_ROW_THRESHOLD && cols > radius * 4 {
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
        let mut row_slice = Vec::with_capacity(cols);
        let mut padded = Vec::with_capacity(cols + 2 * radius);

        for r in 0..rows {
            row_slice.clear();
            row_slice.extend(input.row(r).iter().copied());

            fill_padded_row(&row_slice, radius, &mut padded);

            let out_slice = output.row_mut(r).into_slice().unwrap();
            convolve_1d_padded(&padded, &kernel, out_slice);
        }
    }

    output
}

fn blur_cols<F: Bm3dFloat>(input: ArrayView2<F>, sigma: F) -> Array2<F> {
    let (rows, cols) = input.dim();
    let kernel = gaussian_kernel_1d(sigma);
    let radius = kernel.len() / 2;

    if cols == 0 || rows == 0 {
        return Array2::zeros((rows, cols));
    }

    let mut output = Array2::zeros((rows, cols));

    if cols >= PARALLEL_ROW_THRESHOLD && rows > radius * 4 {
        let col_indices: Vec<usize> = (0..cols).collect();

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

        for (c, col_out) in results.into_iter().enumerate() {
            for (r, &val) in col_out.iter().enumerate() {
                output[[r, c]] = val;
            }
        }
    } else {
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
///
/// sigma_y is applied along axis 0, sigma_x along axis 1.
pub fn gaussian_blur_2d<F: Bm3dFloat>(input: ArrayView2<F>, sigma_y: F, sigma_x: F) -> Array2<F> {
    let blurred_x = blur_rows(input, sigma_x);
    blur_cols(blurred_x.view(), sigma_y)
}

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
        let (_, median, _) = data.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        *median
    } else {
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
pub fn median_axis0<F: Bm3dFloat>(input: ArrayView2<F>) -> Array1<F> {
    let (rows, cols) = input.dim();

    if rows == 0 || cols == 0 {
        return Array1::zeros(cols);
    }

    let mut output = Array1::zeros(cols);

    let mut col_data: Vec<F> = Vec::with_capacity(rows);

    for c in 0..cols {
        col_data.clear();
        col_data.extend((0..rows).map(|r| input[[r, c]]));
        output[c] = median_slice(&mut col_data);
    }

    output
}

/// Estimate the static vertical streak profile.
///
/// # Arguments
/// * `sinogram` - Input 2D sinogram (H x W)
/// * `sigma_smooth` - Sigma for vertical smoothing
/// * `iterations` - Number of refinement iterations
///
/// # Returns
/// 1D streak profile of length W
pub fn estimate_streak_profile_impl<F: Bm3dFloat>(
    sinogram: ArrayView2<F>,
    sigma_smooth: F,
    iterations: usize,
) -> Array1<F> {
    let (rows, cols) = sinogram.dim();

    let mut z_clean = sinogram.to_owned();
    let mut streak_acc = Array1::zeros(cols);

    let horizontal_sigma = F::from_f64_c(STREAK_HORIZONTAL_SIGMA);
    let update_sigma = F::from_f64_c(STREAK_UPDATE_SIGMA);

    for _ in 0..iterations {
        let z_smooth = gaussian_blur_2d(z_clean.view(), sigma_smooth, horizontal_sigma);

        let residual = &z_clean - &z_smooth;

        let streak_update = median_axis0(residual.view());

        let streak_update_smooth = gaussian_blur_1d(streak_update.view(), update_sigma);

        streak_acc += &streak_update_smooth;

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

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

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

    #[test]
    fn test_reflect_index_in_bounds() {
        assert_eq!(reflect_index(0, 5), 0);
        assert_eq!(reflect_index(2, 5), 2);
        assert_eq!(reflect_index(4, 5), 4);
    }

    #[test]
    fn test_reflect_index_negative() {
        assert_eq!(reflect_index(-1, 5), 0);
        assert_eq!(reflect_index(-2, 5), 1);
        assert_eq!(reflect_index(-3, 5), 2);
    }

    #[test]
    fn test_reflect_index_beyond_end() {
        assert_eq!(reflect_index(5, 5), 3);
        assert_eq!(reflect_index(6, 5), 2);
        assert_eq!(reflect_index(7, 5), 1);
    }

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

    #[test]
    fn test_gaussian_1d_identity() {
        let input = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let output = gaussian_blur_1d(input.view(), 0.001f32);
        assert!(
            arrays_approx_equal_1d(&input, &output, 1e-5),
            "Very small sigma should preserve input"
        );
    }

    #[test]
    fn test_gaussian_1d_uniform() {
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
        let mut input = Array1::zeros(20);
        for i in 10..20 {
            input[i] = 1.0f32;
        }
        let output = gaussian_blur_1d(input.view(), 2.0f32);

        assert!(output[9] > 0.0 && output[9] < 1.0, "Should smooth the step");
        assert!(
            output[10] > 0.0 && output[10] < 1.0,
            "Should smooth the step"
        );
    }

    #[test]
    fn test_gaussian_1d_preserves_mean() {
        let input = Array1::from_vec(vec![1.0f32, 3.0, 2.0, 5.0, 4.0, 2.0, 3.0, 1.0]);
        let output = gaussian_blur_1d(input.view(), 1.0f32);

        let input_mean: f32 = input.iter().sum::<f32>() / input.len() as f32;
        let output_mean: f32 = output.iter().sum::<f32>() / output.len() as f32;

        assert!(
            approx_eq(input_mean, output_mean, 0.5),
            "Mean should be approximately preserved: {} vs {}",
            input_mean,
            output_mean
        );
    }

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

    #[test]
    fn test_gaussian_2d_uniform() {
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
        let input = Array2::from_shape_fn((8, 8), |(r, c)| (r * 8 + c) as f32 / 64.0);

        let output_2d = gaussian_blur_2d(input.view(), 1.5f32, 2.0f32);

        let after_rows = blur_rows(input.view(), 2.0f32);
        let after_cols = blur_cols(after_rows.view(), 1.5f32);

        assert!(
            arrays_approx_equal_2d(&output_2d, &after_cols, 1e-6),
            "2D blur should be separable"
        );
    }

    #[test]
    fn test_gaussian_2d_anisotropic() {
        let input = Array2::from_shape_fn(
            (16, 16),
            |(r, c)| {
                if r == 8 && c == 8 { 1.0f32 } else { 0.0 }
            },
        );

        let output_iso = gaussian_blur_2d(input.view(), 2.0f32, 2.0f32);
        let output_aniso = gaussian_blur_2d(input.view(), 4.0f32, 1.0f32);

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

    #[test]
    fn test_median_axis0_simple() {
        let input =
            Array2::from_shape_vec((3, 3), vec![1.0f32, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0])
                .unwrap();

        let result = median_axis0(input.view());

        assert_eq!(result.len(), 3);
        assert!(approx_eq(result[0], 2.0, 1e-6));
        assert!(approx_eq(result[1], 5.0, 1e-6));
        assert!(approx_eq(result[2], 8.0, 1e-6));
    }

    #[test]
    fn test_median_axis0_even_rows() {
        let input =
            Array2::from_shape_vec((4, 2), vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0])
                .unwrap();

        let result = median_axis0(input.view());

        assert_eq!(result.len(), 2);
        assert!(approx_eq(result[0], 2.5, 1e-6));
        assert!(approx_eq(result[1], 25.0, 1e-6));
    }

    #[test]
    fn test_median_axis0_single_row() {
        let input = Array2::from_shape_vec((1, 4), vec![5.0f32, 3.0, 8.0, 1.0]).unwrap();
        let result = median_axis0(input.view());

        assert_eq!(result.len(), 4);
        assert!(approx_eq(result[0], 5.0, 1e-6));
        assert!(approx_eq(result[1], 3.0, 1e-6));
        assert!(approx_eq(result[2], 8.0, 1e-6));
        assert!(approx_eq(result[3], 1.0, 1e-6));
    }

    #[test]
    fn test_median_axis0_unsorted() {
        let input = Array2::from_shape_vec((5, 1), vec![5.0f32, 1.0, 9.0, 3.0, 7.0]).unwrap();
        let result = median_axis0(input.view());

        assert_eq!(result.len(), 1);
        assert!(approx_eq(result[0], 5.0, 1e-6));
    }

    #[test]
    fn test_streak_profile_uniform_image() {
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
        let mut input = Array2::from_elem((32, 64), 0.0f32);
        for r in 0..32 {
            input[[r, 20]] = 1.0;
        }

        let profile = estimate_streak_profile_impl(input.view(), 3.0f32, 3);

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
        let mut input = Array2::from_elem((32, 64), 0.0f32);
        for r in 0..32 {
            input[[r, 10]] = 0.5;
            input[[r, 30]] = 1.0;
            input[[r, 50]] = 0.8;
        }

        let profile = estimate_streak_profile_impl(input.view(), 3.0f32, 3);

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
        let mut input = Array2::from_elem((32, 32), 0.5f32);
        for r in 0..32 {
            input[[r, 16]] = 1.0;
        }

        let profile_1 = estimate_streak_profile_impl(input.view(), 3.0f32, 1);
        let profile_3 = estimate_streak_profile_impl(input.view(), 3.0f32, 3);

        assert!(profile_1[16] > 0.0, "1 iteration should detect streak");
        assert!(profile_3[16] > 0.0, "3 iterations should detect streak");
    }

    #[test]
    fn test_streak_profile_shape() {
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
        let mut input = Array2::from_elem((64, 64), 0.0f32);
        for c in 0..64 {
            input[[32, c]] = 1.0;
        }

        let profile = estimate_streak_profile_impl(input.view(), 3.0f32, 3);

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
        let mut input = Array2::from_elem((32, 32), 0.0f32);
        for r in 0..32 {
            input[[r, 16]] = 1.0;
        }

        let profile = estimate_streak_profile_impl(input.view(), 1.0f32, 3);
        assert!(profile[16] > 0.0, "Should detect streak with small sigma");
    }

    #[test]
    fn test_streak_profile_large_sigma() {
        let mut input = Array2::from_elem((32, 32), 0.0f32);
        for r in 0..32 {
            input[[r, 16]] = 1.0;
        }

        let profile_small = estimate_streak_profile_impl(input.view(), 1.0f32, 3);
        let profile_large = estimate_streak_profile_impl(input.view(), 10.0f32, 3);

        let peak_small = profile_small[16];
        let peak_large = profile_large[16];

        assert!(peak_small > 0.0, "Small sigma should detect");
        assert!(peak_large > 0.0, "Large sigma should detect");
    }
}
