//! Criterion benchmarks for BM3D core operations.
//!
//! Run with: cargo bench
//! Run specific: cargo bench -- bench_fft2d
//!
//! Note: This benchmark file includes its own implementations because the main
//! crate is cdylib-only (PyO3 extension) and can't be linked as rlib.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, s};
use rand::prelude::*;
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::Arc;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

// =============================================================================
// Constants (duplicated from main crate for standalone benchmarks)
// =============================================================================

const WIENER_EPSILON: f32 = 1e-8;
const MAX_WIENER_WEIGHT: f32 = 1e6;
const AGGREGATION_EPSILON: f32 = 1e-6;
const RAYON_MIN_CHUNK_LEN: usize = 2048;
const HADAMARD_PATCH_SIZE: usize = 8;
const GAUSSIAN_TRUNCATE: f32 = 4.0;
const STREAK_HORIZONTAL_SIGMA: f32 = 3.0;
const STREAK_UPDATE_SIGMA: f32 = 1.0;

// =============================================================================
// Core Implementations (duplicated for standalone benchmarks)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bm3dMode {
    HardThreshold,
    Wiener,
}

#[derive(Clone, Copy)]
pub struct PatchMatch {
    pub row: usize,
    pub col: usize,
    pub distance: f32,
}

impl PartialEq for PatchMatch {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for PatchMatch {}

impl PartialOrd for PatchMatch {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PatchMatch {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
    }
}

// FFT functions
pub fn fft2d(
    input: ArrayView2<f32>,
    fft_row_plan: &Arc<dyn Fft<f32>>,
    fft_col_plan: &Arc<dyn Fft<f32>>,
) -> Array2<Complex<f32>> {
    let (rows, cols) = input.dim();
    let mut buffer: Vec<Complex<f32>> = input.iter().map(|&v| Complex::new(v, 0.0)).collect();

    // Row-wise FFT
    for r in 0..rows {
        let start = r * cols;
        fft_row_plan.process(&mut buffer[start..start + cols]);
    }

    // Column-wise FFT
    let mut col_buf = vec![Complex::new(0.0, 0.0); rows];
    for c in 0..cols {
        for r in 0..rows {
            col_buf[r] = buffer[r * cols + c];
        }
        fft_col_plan.process(&mut col_buf);
        for r in 0..rows {
            buffer[r * cols + c] = col_buf[r];
        }
    }

    Array2::from_shape_vec((rows, cols), buffer).unwrap()
}

pub fn ifft2d(
    input: &Array2<Complex<f32>>,
    ifft_row_plan: &Arc<dyn Fft<f32>>,
    ifft_col_plan: &Arc<dyn Fft<f32>>,
) -> Array2<f32> {
    let (rows, cols) = input.dim();
    let scale = 1.0 / (rows * cols) as f32;
    let mut buffer: Vec<Complex<f32>> = input.iter().cloned().collect();

    // Row-wise IFFT
    for r in 0..rows {
        let start = r * cols;
        ifft_row_plan.process(&mut buffer[start..start + cols]);
    }

    // Column-wise IFFT
    let mut col_buf = vec![Complex::new(0.0, 0.0); rows];
    for c in 0..cols {
        for r in 0..rows {
            col_buf[r] = buffer[r * cols + c];
        }
        ifft_col_plan.process(&mut col_buf);
        for r in 0..rows {
            buffer[r * cols + c] = col_buf[r];
        }
    }

    Array2::from_shape_fn((rows, cols), |(r, c)| buffer[r * cols + c].re * scale)
}

// WHT functions
fn fwht8(buf: &mut [f32; 8]) {
    // Radix-2 in-place
    for h in [1, 2, 4] {
        for i in (0..8).step_by(h * 2) {
            for j in 0..h {
                let x = buf[i + j];
                let y = buf[i + j + h];
                buf[i + j] = x + y;
                buf[i + j + h] = x - y;
            }
        }
    }
}

pub fn wht2d_8x8_forward(input: ArrayView2<f32>) -> Array2<Complex<f32>> {
    let mut data = [0.0f32; 64];
    let mut idx = 0;
    for r in 0..8 {
        for c in 0..8 {
            data[idx] = input[[r, c]];
            idx += 1;
        }
    }
    // Row transforms
    for r in 0..8 {
        let mut row_buf = [0.0f32; 8];
        for c in 0..8 {
            row_buf[c] = data[r * 8 + c];
        }
        fwht8(&mut row_buf);
        for c in 0..8 {
            data[r * 8 + c] = row_buf[c];
        }
    }
    // Column transforms
    for c in 0..8 {
        let mut col_buf = [0.0f32; 8];
        for r in 0..8 {
            col_buf[r] = data[r * 8 + c];
        }
        fwht8(&mut col_buf);
        for r in 0..8 {
            data[r * 8 + c] = col_buf[r];
        }
    }
    let mut output = Array2::<Complex<f32>>::zeros((8, 8));
    idx = 0;
    for r in 0..8 {
        for c in 0..8 {
            output[[r, c]] = Complex::new(data[idx], 0.0);
            idx += 1;
        }
    }
    output
}

pub fn wht2d_8x8_inverse(input: &Array2<Complex<f32>>) -> Array2<f32> {
    let mut data = [0.0f32; 64];
    let mut idx = 0;
    for r in 0..8 {
        for c in 0..8 {
            data[idx] = input[[r, c]].re;
            idx += 1;
        }
    }
    for r in 0..8 {
        let mut row_buf = [0.0f32; 8];
        for c in 0..8 {
            row_buf[c] = data[r * 8 + c];
        }
        fwht8(&mut row_buf);
        for c in 0..8 {
            data[r * 8 + c] = row_buf[c];
        }
    }
    for c in 0..8 {
        let mut col_buf = [0.0f32; 8];
        for r in 0..8 {
            col_buf[r] = data[r * 8 + c];
        }
        fwht8(&mut col_buf);
        for r in 0..8 {
            data[r * 8 + c] = col_buf[r];
        }
    }
    let mut output = Array2::<f32>::zeros((8, 8));
    idx = 0;
    let scale = 1.0 / 64.0;
    for r in 0..8 {
        for c in 0..8 {
            output[[r, c]] = data[idx] * scale;
            idx += 1;
        }
    }
    output
}

// Block matching
pub fn compute_integral_images(image: ArrayView2<f32>) -> (Array2<f32>, Array2<f32>) {
    let (rows, cols) = image.dim();
    let mut integral_sum = Array2::<f32>::zeros((rows + 1, cols + 1));
    let mut integral_sq = Array2::<f32>::zeros((rows + 1, cols + 1));

    for r in 0..rows {
        for c in 0..cols {
            let val = image[[r, c]];
            integral_sum[[r + 1, c + 1]] = val + integral_sum[[r, c + 1]] + integral_sum[[r + 1, c]] - integral_sum[[r, c]];
            integral_sq[[r + 1, c + 1]] = val * val + integral_sq[[r, c + 1]] + integral_sq[[r + 1, c]] - integral_sq[[r, c]];
        }
    }
    (integral_sum, integral_sq)
}

fn get_patch_sums(
    integral: &Array2<f32>,
    row: usize,
    col: usize,
    patch_h: usize,
    patch_w: usize,
) -> f32 {
    let r1 = row;
    let c1 = col;
    let r2 = row + patch_h;
    let c2 = col + patch_w;
    integral[[r2, c2]] - integral[[r1, c2]] - integral[[r2, c1]] + integral[[r1, c1]]
}

pub fn find_similar_patches(
    image: ArrayView2<f32>,
    integral_sum: &Array2<f32>,
    integral_sq_sum: &Array2<f32>,
    ref_pos: (usize, usize),
    patch_size: (usize, usize),
    search_window: (usize, usize),
    max_matches: usize,
    step_size: usize,
) -> Vec<PatchMatch> {
    let (rows, cols) = image.dim();
    let (ref_r, ref_c) = ref_pos;
    let (ph, pw) = patch_size;
    let (sw_h, sw_w) = search_window;

    let r_start = ref_r.saturating_sub(sw_h / 2);
    let c_start = ref_c.saturating_sub(sw_w / 2);
    let r_end = (ref_r + sw_h / 2 + 1).min(rows - ph + 1);
    let c_end = (ref_c + sw_w / 2 + 1).min(cols - pw + 1);

    let n = ph * pw;
    let ref_sum = get_patch_sums(integral_sum, ref_r, ref_c, ph, pw);
    let ref_sq_sum = get_patch_sums(integral_sq_sum, ref_r, ref_c, ph, pw);

    let mut heap: BinaryHeap<PatchMatch> = BinaryHeap::new();

    for r in (r_start..r_end).step_by(step_size) {
        for c in (c_start..c_end).step_by(step_size) {
            let cand_sum = get_patch_sums(integral_sum, r, c, ph, pw);
            let cand_sq_sum = get_patch_sums(integral_sq_sum, r, c, ph, pw);

            let distance = (ref_sq_sum + cand_sq_sum - 2.0 * ref_sum * cand_sum / n as f32) / n as f32;
            let distance = distance.max(0.0);

            if heap.len() < max_matches {
                heap.push(PatchMatch { row: r, col: c, distance });
            } else if let Some(worst) = heap.peek() {
                if distance < worst.distance {
                    heap.pop();
                    heap.push(PatchMatch { row: r, col: c, distance });
                }
            }
        }
    }

    let mut result: Vec<_> = heap.into_iter().collect();
    result.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
    result
}

// Streak profile estimation
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
        kernel[i] = (-0.5 * x * x / sigma2).exp();
        sum += kernel[i];
    }
    for k in kernel.iter_mut() {
        *k /= sum;
    }
    kernel
}

fn reflect_index(idx: isize, len: usize) -> usize {
    if idx < 0 {
        (-idx - 1) as usize
    } else if idx >= len as isize {
        2 * len - idx as usize - 1
    } else {
        idx as usize
    }
}

pub fn gaussian_blur_1d(input: ArrayView1<f32>, sigma: f32) -> Array1<f32> {
    let kernel = gaussian_kernel_1d(sigma);
    let radius = kernel.len() / 2;
    let len = input.len();
    let mut output = Array1::zeros(len);
    for i in 0..len {
        let mut sum = 0.0;
        for (ki, &kval) in kernel.iter().enumerate() {
            let src_idx = i as isize + ki as isize - radius as isize;
            let src_idx = reflect_index(src_idx, len);
            sum += input[src_idx] * kval;
        }
        output[i] = sum;
    }
    output
}

fn blur_rows(input: ArrayView2<f32>, sigma: f32) -> Array2<f32> {
    let kernel = gaussian_kernel_1d(sigma);
    let radius = kernel.len() / 2;
    let (rows, cols) = input.dim();
    let mut output = Array2::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let mut sum = 0.0;
            for (ki, &kval) in kernel.iter().enumerate() {
                let src_r = r as isize + ki as isize - radius as isize;
                let src_r = reflect_index(src_r, rows);
                sum += input[[src_r, c]] * kval;
            }
            output[[r, c]] = sum;
        }
    }
    output
}

fn blur_cols(input: ArrayView2<f32>, sigma: f32) -> Array2<f32> {
    let kernel = gaussian_kernel_1d(sigma);
    let radius = kernel.len() / 2;
    let (rows, cols) = input.dim();
    let mut output = Array2::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let mut sum = 0.0;
            for (ki, &kval) in kernel.iter().enumerate() {
                let src_c = c as isize + ki as isize - radius as isize;
                let src_c = reflect_index(src_c, cols);
                sum += input[[r, src_c]] * kval;
            }
            output[[r, c]] = sum;
        }
    }
    output
}

pub fn gaussian_blur_2d(input: ArrayView2<f32>, sigma_y: f32, sigma_x: f32) -> Array2<f32> {
    let blurred_rows = blur_rows(input, sigma_y);
    blur_cols(blurred_rows.view(), sigma_x)
}

fn median_slice(data: &mut [f32]) -> f32 {
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let len = data.len();
    if len % 2 == 0 {
        (data[len / 2 - 1] + data[len / 2]) / 2.0
    } else {
        data[len / 2]
    }
}

pub fn median_axis0(input: ArrayView2<f32>) -> Array1<f32> {
    let (rows, cols) = input.dim();
    let mut result = Array1::zeros(cols);
    for c in 0..cols {
        let mut col_data: Vec<f32> = (0..rows).map(|r| input[[r, c]]).collect();
        result[c] = median_slice(&mut col_data);
    }
    result
}

pub fn estimate_streak_profile_impl(
    sinogram: ArrayView2<f32>,
    sigma_smooth: f32,
    iterations: usize,
) -> Array1<f32> {
    let (rows, cols) = sinogram.dim();
    let mut z_clean = sinogram.to_owned();
    let mut streak_acc = Array1::zeros(cols);

    for _ in 0..iterations {
        let z_smooth = gaussian_blur_2d(z_clean.view(), sigma_smooth, STREAK_HORIZONTAL_SIGMA);
        let residual = &z_clean - &z_smooth;
        let streak_update = median_axis0(residual.view());
        let streak_update_smooth = gaussian_blur_1d(streak_update.view(), STREAK_UPDATE_SIGMA);
        streak_acc = streak_acc + &streak_update_smooth;
        for r in 0..rows {
            for c in 0..cols {
                z_clean[[r, c]] -= streak_update_smooth[c];
            }
        }
    }
    streak_acc
}

// BM3D Plans
pub struct Bm3dPlans {
    fft_2d_row: Arc<dyn Fft<f32>>,
    fft_2d_col: Arc<dyn Fft<f32>>,
    ifft_2d_row: Arc<dyn Fft<f32>>,
    ifft_2d_col: Arc<dyn Fft<f32>>,
    fft_1d_plans: Vec<Arc<dyn Fft<f32>>>,
    ifft_1d_plans: Vec<Arc<dyn Fft<f32>>>,
}

impl Bm3dPlans {
    pub fn new(patch_size: usize, max_matches: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft_2d_row = planner.plan_fft_forward(patch_size);
        let fft_2d_col = planner.plan_fft_forward(patch_size);
        let ifft_2d_row = planner.plan_fft_inverse(patch_size);
        let ifft_2d_col = planner.plan_fft_inverse(patch_size);

        let mut fft_1d_plans = Vec::with_capacity(max_matches + 1);
        let mut ifft_1d_plans = Vec::with_capacity(max_matches + 1);

        fft_1d_plans.push(planner.plan_fft_forward(1));
        ifft_1d_plans.push(planner.plan_fft_inverse(1));

        for k in 1..=max_matches {
            fft_1d_plans.push(planner.plan_fft_forward(k));
            ifft_1d_plans.push(planner.plan_fft_inverse(k));
        }

        Self {
            fft_2d_row, fft_2d_col, ifft_2d_row, ifft_2d_col,
            fft_1d_plans, ifft_1d_plans,
        }
    }
}

// Simplified BM3D kernel for benchmarking
pub fn run_bm3d_kernel(
    input_noisy: ArrayView2<f32>,
    input_pilot: ArrayView2<f32>,
    mode: Bm3dMode,
    sigma_psd: ArrayView2<f32>,
    sigma_map: ArrayView2<f32>,
    sigma_random: f32,
    threshold: f32,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
    plans: &Bm3dPlans,
) -> Array2<f32> {
    let (rows, cols) = input_noisy.dim();
    let use_sigma_map = sigma_map.dim() == (rows, cols);
    let use_colored_noise = sigma_psd.dim() == (patch_size, patch_size);
    let scalar_sigma_sq = sigma_random * sigma_random;
    let use_hadamard = patch_size == HADAMARD_PATCH_SIZE;

    let (integral_sum, integral_sq_sum) = compute_integral_images(input_pilot);

    let mut ref_coords = Vec::new();
    let r_end = rows.saturating_sub(patch_size) + 1;
    let c_end = cols.saturating_sub(patch_size) + 1;

    for r in (0..r_end).step_by(step_size) {
        for c in (0..c_end).step_by(step_size) {
            ref_coords.push((r, c));
        }
    }

    let fft_2d_row_ref = &plans.fft_2d_row;
    let fft_2d_col_ref = &plans.fft_2d_col;
    let ifft_2d_row_ref = &plans.ifft_2d_row;
    let ifft_2d_col_ref = &plans.ifft_2d_col;
    let fft_1d_plans_ref = &plans.fft_1d_plans;
    let ifft_1d_plans_ref = &plans.ifft_1d_plans;

    let (final_num, final_den) = ref_coords.par_iter()
        .with_min_len(RAYON_MIN_CHUNK_LEN)
        .fold(
            || (Array2::<f32>::zeros((rows, cols)), Array2::<f32>::zeros((rows, cols))),
            |mut acc, &(ref_r, ref_c)| {
                let (numerator_acc, denominator_acc) = &mut acc;

                let matches = find_similar_patches(
                    input_pilot,
                    &integral_sum,
                    &integral_sq_sum,
                    (ref_r, ref_c),
                    (patch_size, patch_size),
                    (search_window, search_window),
                    max_matches,
                    step_size,
                );
                let k = matches.len();
                if k == 0 { return acc; }

                let local_sigma_streak = if use_sigma_map { sigma_map[[ref_r, ref_c]] } else { 0.0 };

                // Build patch groups
                let mut group_noisy = Array3::<f32>::zeros((k, patch_size, patch_size));
                let mut group_pilot = Array3::<f32>::zeros((k, patch_size, patch_size));
                for (idx, m) in matches.iter().enumerate() {
                    let p_n = input_noisy.slice(s![m.row..m.row+patch_size, m.col..m.col+patch_size]);
                    let p_p = input_pilot.slice(s![m.row..m.row+patch_size, m.col..m.col+patch_size]);
                    group_noisy.slice_mut(s![idx, .., ..]).assign(&p_n);
                    group_pilot.slice_mut(s![idx, .., ..]).assign(&p_p);
                }

                // Forward 2D transforms
                let mut g_noisy_c = Array3::<Complex<f32>>::zeros((k, patch_size, patch_size));
                let mut g_pilot_c = Array3::<Complex<f32>>::zeros((k, patch_size, patch_size));

                for i in 0..k {
                    let slice_n = group_noisy.slice(s![i, .., ..]);
                    if use_hadamard {
                        let fft_n = wht2d_8x8_forward(slice_n);
                        g_noisy_c.slice_mut(s![i, .., ..]).assign(&fft_n);
                    } else {
                        let fft_n = fft2d(slice_n, fft_2d_row_ref, fft_2d_col_ref);
                        g_noisy_c.slice_mut(s![i, .., ..]).assign(&fft_n);
                    }
                    if mode == Bm3dMode::Wiener {
                        let slice_p = group_pilot.slice(s![i, .., ..]);
                        if use_hadamard {
                            let fft_p = wht2d_8x8_forward(slice_p);
                            g_pilot_c.slice_mut(s![i, .., ..]).assign(&fft_p);
                        } else {
                            let fft_p = fft2d(slice_p, fft_2d_row_ref, fft_2d_col_ref);
                            g_pilot_c.slice_mut(s![i, .., ..]).assign(&fft_p);
                        }
                    }
                }

                // Forward 1D transforms
                let fft_k_plan = &fft_1d_plans_ref[k];
                for r in 0..patch_size {
                    for c in 0..patch_size {
                        let mut vec_n: Vec<Complex<f32>> = (0..k).map(|i| g_noisy_c[[i, r, c]]).collect();
                        fft_k_plan.process(&mut vec_n);
                        for i in 0..k { g_noisy_c[[i, r, c]] = vec_n[i]; }
                        if mode == Bm3dMode::Wiener {
                            let mut vec_p: Vec<Complex<f32>> = (0..k).map(|i| g_pilot_c[[i, r, c]]).collect();
                            fft_k_plan.process(&mut vec_p);
                            for i in 0..k { g_pilot_c[[i, r, c]] = vec_p[i]; }
                        }
                    }
                }

                // Filtering
                let mut weight_g = 1.0;
                let spatial_scale = patch_size as f32;
                let spatial_scale_sq = spatial_scale * spatial_scale;

                match mode {
                    Bm3dMode::HardThreshold => {
                        let mut nz_count = 0;
                        for i in 0..k {
                            for r in 0..patch_size {
                                for c in 0..patch_size {
                                    let coeff = g_noisy_c[[i, r, c]];
                                    let sigma_s_dist = if use_colored_noise { sigma_psd[[r, c]] } else { 0.0 };
                                    let effective_sigma_s = sigma_s_dist * local_sigma_streak;
                                    let var_r = k as f32 * scalar_sigma_sq;
                                    let var_s = (k * k) as f32 * effective_sigma_s * effective_sigma_s;
                                    let noise_std_coeff = (var_r + var_s).sqrt() * spatial_scale;
                                    if coeff.norm() < threshold * noise_std_coeff {
                                        g_noisy_c[[i, r, c]] = Complex::new(0.0, 0.0);
                                    } else {
                                        nz_count += 1;
                                    }
                                }
                            }
                        }
                        if nz_count > 0 { weight_g = 1.0 / (nz_count as f32 + 1.0); }
                    }
                    Bm3dMode::Wiener => {
                        let mut wiener_sum = 0.0;
                        for i in 0..k {
                            for r in 0..patch_size {
                                for c in 0..patch_size {
                                    let p_val = g_pilot_c[[i, r, c]];
                                    let n_val = g_noisy_c[[i, r, c]];
                                    let sigma_s_dist = if use_colored_noise { sigma_psd[[r, c]] } else { 0.0 };
                                    let effective_sigma_s = sigma_s_dist * local_sigma_streak;
                                    let var_r = k as f32 * scalar_sigma_sq;
                                    let var_s = (k * k) as f32 * effective_sigma_s * effective_sigma_s;
                                    let noise_var_coeff = (var_r + var_s) * spatial_scale_sq;
                                    let w = p_val.norm_sqr() / (p_val.norm_sqr() + noise_var_coeff + WIENER_EPSILON);
                                    g_noisy_c[[i, r, c]] = n_val * w;
                                    wiener_sum += w * w;
                                }
                            }
                        }
                        weight_g = 1.0 / (wiener_sum * scalar_sigma_sq + WIENER_EPSILON);
                        if weight_g > MAX_WIENER_WEIGHT { weight_g = MAX_WIENER_WEIGHT; }
                    }
                }

                // Inverse 1D transforms
                let ifft_k_plan = &ifft_1d_plans_ref[k];
                let norm_k = 1.0 / k as f32;
                for r in 0..patch_size {
                    for c in 0..patch_size {
                        let mut vec: Vec<Complex<f32>> = (0..k).map(|i| g_noisy_c[[i, r, c]]).collect();
                        ifft_k_plan.process(&mut vec);
                        for i in 0..k { g_noisy_c[[i, r, c]] = vec[i] * norm_k; }
                    }
                }

                // Inverse 2D transforms and aggregation
                for i in 0..k {
                    let complex_slice = g_noisy_c.slice(s![i, .., ..]).to_owned();
                    let spatial = if use_hadamard {
                        wht2d_8x8_inverse(&complex_slice)
                    } else {
                        ifft2d(&complex_slice, ifft_2d_row_ref, ifft_2d_col_ref)
                    };
                    let m = &matches[i];
                    for pr in 0..patch_size {
                        for pc in 0..patch_size {
                            let tr = m.row + pr;
                            let tc = m.col + pc;
                            if tr < rows && tc < cols {
                                numerator_acc[[tr, tc]] += spatial[[pr, pc]] * weight_g;
                                denominator_acc[[tr, tc]] += weight_g;
                            }
                        }
                    }
                }
                acc
            })
        .reduce(
            || (Array2::<f32>::zeros((rows, cols)), Array2::<f32>::zeros((rows, cols))),
            |mut a, b| { a.0 += &b.0; a.1 += &b.1; a }
        );

    // Finalize
    let mut output = Array2::<f32>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let num = final_num[[r, c]];
            let den = final_den[[r, c]];
            if den > AGGREGATION_EPSILON {
                output[[r, c]] = num / den;
            } else {
                output[[r, c]] = input_noisy[[r, c]];
            }
        }
    }
    output
}

// =============================================================================
// Test Data Generation Helpers
// =============================================================================

fn random_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_fn((rows, cols), |_| rng.gen::<f32>())
}

fn sinogram_like(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Array2::from_shape_fn((rows, cols), |(r, c)| {
        let base = ((r as f32 / rows as f32) * std::f32::consts::PI).sin();
        let col_mod = ((c as f32 / cols as f32) * 4.0 * std::f32::consts::PI).cos() * 0.1;
        base + col_mod
    });
    for &streak_col in &[cols / 4, cols / 2, 3 * cols / 4] {
        for r in 0..rows {
            data[[r, streak_col]] += 0.3;
        }
    }
    for r in 0..rows {
        for c in 0..cols {
            data[[r, c]] += rng.gen::<f32>() * 0.05;
        }
    }
    data
}

// =============================================================================
// Transform Benchmarks
// =============================================================================

fn bench_fft2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft2d");

    for size in [64, 256] {
        let input = random_matrix(size, size, 42);
        let mut planner = FftPlanner::new();
        let fft_row = planner.plan_fft_forward(size);
        let fft_col = planner.plan_fft_forward(size);
        let ifft_row = planner.plan_fft_inverse(size);
        let ifft_col = planner.plan_fft_inverse(size);

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(
            BenchmarkId::new("forward", format!("{}x{}", size, size)),
            &size,
            |b, _| {
                b.iter(|| fft2d(black_box(input.view()), &fft_row, &fft_col))
            },
        );

        let freq = fft2d(input.view(), &fft_row, &fft_col);
        group.bench_with_input(
            BenchmarkId::new("inverse", format!("{}x{}", size, size)),
            &size,
            |b, _| {
                b.iter(|| ifft2d(black_box(&freq), &ifft_row, &ifft_col))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("roundtrip", format!("{}x{}", size, size)),
            &size,
            |b, _| {
                b.iter(|| {
                    let freq = fft2d(black_box(input.view()), &fft_row, &fft_col);
                    ifft2d(&freq, &ifft_row, &ifft_col)
                })
            },
        );
    }

    group.finish();
}

fn bench_wht_8x8(c: &mut Criterion) {
    let mut group = c.benchmark_group("wht_8x8");

    let input = random_matrix(8, 8, 42);
    group.throughput(Throughput::Elements(64));

    group.bench_function("forward", |b| {
        b.iter(|| wht2d_8x8_forward(black_box(input.view())))
    });

    let freq = wht2d_8x8_forward(input.view());
    group.bench_function("inverse", |b| {
        b.iter(|| wht2d_8x8_inverse(black_box(&freq)))
    });

    group.bench_function("roundtrip", |b| {
        b.iter(|| {
            let freq = wht2d_8x8_forward(black_box(input.view()));
            wht2d_8x8_inverse(&freq)
        })
    });

    group.finish();
}

// =============================================================================
// Block Matching Benchmarks
// =============================================================================

fn bench_block_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_matching");

    let image = random_matrix(256, 256, 42);
    let (integral_sum, integral_sq_sum) = compute_integral_images(image.view());

    group.bench_function("integral_images_256x256", |b| {
        b.iter(|| compute_integral_images(black_box(image.view())))
    });

    for (patch_size, search_window, max_matches) in [
        (8, 24, 16),
        (8, 24, 32),
        (8, 40, 64),
        (7, 40, 64),
    ] {
        let label = format!("p{}s{}m{}", patch_size, search_window, max_matches);

        group.bench_with_input(
            BenchmarkId::new("find_similar", &label),
            &(patch_size, search_window, max_matches),
            |b, &(ps, sw, mm)| {
                b.iter(|| {
                    find_similar_patches(
                        black_box(image.view()),
                        &integral_sum,
                        &integral_sq_sum,
                        (128, 128),
                        (ps, ps),
                        (sw, sw),
                        mm,
                        2,
                    )
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Streak Profile Estimation Benchmarks
// =============================================================================

fn bench_streak_profile(c: &mut Criterion) {
    let mut group = c.benchmark_group("streak_profile");

    let sinogram = sinogram_like(512, 640, 42);
    group.throughput(Throughput::Elements((512 * 640) as u64));

    for iterations in [1, 3] {
        group.bench_with_input(
            BenchmarkId::new("estimate", format!("512x640_iter{}", iterations)),
            &iterations,
            |b, &iters| {
                b.iter(|| {
                    estimate_streak_profile_impl(
                        black_box(sinogram.view()),
                        3.0,
                        iters,
                    )
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Full BM3D Pipeline Benchmarks
// =============================================================================

fn bench_bm3d_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("bm3d_full");
    group.sample_size(10);

    for (size, label) in [(128, "small_128"), (256, "medium_256")] {
        let image = random_matrix(size, size, 42);
        let sigma_psd = Array2::<f32>::zeros((8, 8));
        let sigma_map = Array2::<f32>::zeros((1, 1));
        let plans = Bm3dPlans::new(8, 64);

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(
            BenchmarkId::new("hard_threshold", label),
            &size,
            |b, _| {
                b.iter(|| {
                    run_bm3d_kernel(
                        black_box(image.view()),
                        image.view(),
                        Bm3dMode::HardThreshold,
                        sigma_psd.view(),
                        sigma_map.view(),
                        0.1,
                        2.7,
                        8,
                        2,
                        24,
                        64,
                        &plans,
                    )
                })
            },
        );

        let pilot = run_bm3d_kernel(
            image.view(),
            image.view(),
            Bm3dMode::HardThreshold,
            sigma_psd.view(),
            sigma_map.view(),
            0.1,
            2.7,
            8,
            2,
            24,
            64,
            &plans,
        );

        group.bench_with_input(
            BenchmarkId::new("wiener", label),
            &size,
            |b, _| {
                b.iter(|| {
                    run_bm3d_kernel(
                        black_box(image.view()),
                        pilot.view(),
                        Bm3dMode::Wiener,
                        sigma_psd.view(),
                        sigma_map.view(),
                        0.1,
                        0.0,
                        8,
                        2,
                        24,
                        64,
                        &plans,
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("full_2pass", label),
            &size,
            |b, _| {
                b.iter(|| {
                    let ht_result = run_bm3d_kernel(
                        black_box(image.view()),
                        image.view(),
                        Bm3dMode::HardThreshold,
                        sigma_psd.view(),
                        sigma_map.view(),
                        0.1,
                        2.7,
                        8,
                        2,
                        24,
                        64,
                        &plans,
                    );
                    run_bm3d_kernel(
                        image.view(),
                        ht_result.view(),
                        Bm3dMode::Wiener,
                        sigma_psd.view(),
                        sigma_map.view(),
                        0.1,
                        0.0,
                        8,
                        2,
                        24,
                        64,
                        &plans,
                    )
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    benches,
    bench_fft2d,
    bench_wht_8x8,
    bench_block_matching,
    bench_streak_profile,
    bench_bm3d_full,
);

criterion_main!(benches);
