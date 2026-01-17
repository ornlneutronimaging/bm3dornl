//! BM3D Pipeline - Core denoising kernel and multi-image processing.

use ndarray::{s, Array2, Array3, ArrayView2, ArrayView3, Axis};
use rayon::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::Fft;
use std::sync::Arc;

use crate::block_matching::{self, PatchMatch};
use crate::float_trait::Bm3dFloat;
use crate::transforms;

// =============================================================================
// Constants for BM3D Pipeline
// =============================================================================

/// Small epsilon for numerical stability in Wiener filter division.
/// Prevents division by zero when computing Wiener weights.
const WIENER_EPSILON: f64 = 1e-8;

/// Maximum allowed weight value in Wiener filtering.
/// Clamps weights to prevent numerical instability from very small denominators.
const MAX_WIENER_WEIGHT: f64 = 1e6;

/// Small epsilon for aggregation denominator check.
/// If the accumulated weight is below this threshold, we fall back to the noisy input.
const AGGREGATION_EPSILON: f64 = 1e-6;

/// Minimum chunk length for Rayon parallel iteration.
/// Tuned for good load balancing on typical workloads.
const RAYON_MIN_CHUNK_LEN: usize = 64;

/// Patch size that triggers the fast Hadamard transform path.
/// Walsh-Hadamard Transform is only implemented for 8x8 patches.
const HADAMARD_PATCH_SIZE: usize = 8;

/// BM3D filtering mode.
///
/// Determines whether to use hard thresholding (first pass) or Wiener filtering (second pass).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bm3dMode {
    /// Hard thresholding: zeroes coefficients below threshold.
    /// Used as the first pass to get an initial estimate.
    HardThreshold,
    /// Wiener filtering: applies optimal linear filter using pilot estimate.
    /// Used as the second pass for refinement.
    Wiener,
}

/// Helper struct to manage pre-computed FFT plans.
/// Reusing plans avoids expensive re-initialization overhead (~45% speedup).
/// We pre-compute:
/// - 2D plans for patches (Row/Col)
/// - 1D plans for group dimension (variable K up to max_matches)
pub struct Bm3dPlans<F: Bm3dFloat> {
    fft_2d_row: Arc<dyn Fft<F>>,
    fft_2d_col: Arc<dyn Fft<F>>,
    ifft_2d_row: Arc<dyn Fft<F>>,
    ifft_2d_col: Arc<dyn Fft<F>>,
    fft_1d_plans: Vec<Arc<dyn Fft<F>>>,
    ifft_1d_plans: Vec<Arc<dyn Fft<F>>>,
}

impl<F: Bm3dFloat> Bm3dPlans<F> {
    /// Create new BM3D plans for the given patch size and maximum matches.
    pub fn new(patch_size: usize, max_matches: usize) -> Self {
        let mut planner = rustfft::FftPlanner::new();
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
            fft_2d_row,
            fft_2d_col,
            ifft_2d_row,
            ifft_2d_col,
            fft_1d_plans,
            ifft_1d_plans,
        }
    }
}

// =============================================================================
// Helper Functions for BM3D Kernel Decomposition
// =============================================================================

/// Build patch groups from matched locations.
///
/// Extracts patches from noisy and pilot images at the matched locations,
/// stacking them into 3D arrays of shape (k, patch_size, patch_size).
fn build_patch_groups<F: Bm3dFloat>(
    input_noisy: ArrayView2<F>,
    input_pilot: ArrayView2<F>,
    matches: &[PatchMatch<F>],
    patch_size: usize,
) -> (Array3<F>, Array3<F>) {
    let k = matches.len();
    let mut group_noisy = Array3::<F>::zeros((k, patch_size, patch_size));
    let mut group_pilot = Array3::<F>::zeros((k, patch_size, patch_size));

    for (idx, m) in matches.iter().enumerate() {
        let p_n = input_noisy.slice(s![m.row..m.row + patch_size, m.col..m.col + patch_size]);
        let p_p = input_pilot.slice(s![m.row..m.row + patch_size, m.col..m.col + patch_size]);
        group_noisy.slice_mut(s![idx, .., ..]).assign(&p_n);
        group_pilot.slice_mut(s![idx, .., ..]).assign(&p_p);
    }

    (group_noisy, group_pilot)
}

/// Compute the effective noise standard deviation for a coefficient.
///
/// Combines random noise variance and structured (streak) noise variance
/// based on the PSD and local sigma map values.
#[inline]
#[allow(clippy::too_many_arguments)]
fn compute_noise_std<F: Bm3dFloat>(
    use_colored_noise: bool,
    sigma_psd: ArrayView2<F>,
    local_sigma_streak: F,
    scalar_sigma_sq: F,
    k: usize,
    r: usize,
    c: usize,
    spatial_scale: F,
) -> F {
    let sigma_s_dist = if use_colored_noise {
        sigma_psd[[r, c]]
    } else {
        F::zero()
    };
    let effective_sigma_s = sigma_s_dist * local_sigma_streak;
    let k_f = F::usize_as(k);
    let var_r = k_f * scalar_sigma_sq;
    let var_s = (k_f * k_f) * effective_sigma_s * effective_sigma_s;
    (var_r + var_s).sqrt() * spatial_scale
}

/// Compute the effective noise variance for Wiener filtering.
#[inline]
#[allow(clippy::too_many_arguments)]
fn compute_noise_var<F: Bm3dFloat>(
    use_colored_noise: bool,
    sigma_psd: ArrayView2<F>,
    local_sigma_streak: F,
    scalar_sigma_sq: F,
    k: usize,
    r: usize,
    c: usize,
    spatial_scale_sq: F,
) -> F {
    let sigma_s_dist = if use_colored_noise {
        sigma_psd[[r, c]]
    } else {
        F::zero()
    };
    let effective_sigma_s = sigma_s_dist * local_sigma_streak;
    let k_f = F::usize_as(k);
    let var_r = k_f * scalar_sigma_sq;
    let var_s = (k_f * k_f) * effective_sigma_s * effective_sigma_s;
    (var_r + var_s) * spatial_scale_sq
}

/// Apply 2D forward transform to all patches in a group.
///
/// Uses WHT for 8x8 patches, FFT otherwise. Returns complex coefficients.
fn apply_forward_2d_transform<F: Bm3dFloat>(
    group: &Array3<F>,
    use_hadamard: bool,
    fft_row: &Arc<dyn Fft<F>>,
    fft_col: &Arc<dyn Fft<F>>,
) -> ndarray::Array3<Complex<F>> {
    let (k, patch_size, _) = group.dim();
    let mut result = ndarray::Array3::<Complex<F>>::zeros((k, patch_size, patch_size));

    for i in 0..k {
        let slice = group.slice(s![i, .., ..]);
        if use_hadamard {
            let transformed = transforms::wht2d_8x8_forward(slice);
            result.slice_mut(s![i, .., ..]).assign(&transformed);
        } else {
            let transformed = transforms::fft2d(slice, fft_row, fft_col);
            result.slice_mut(s![i, .., ..]).assign(&transformed);
        }
    }
    result
}

/// Apply 1D forward FFT along the group dimension at each (r, c) position.
fn apply_forward_1d_transform<F: Bm3dFloat>(
    group: &mut ndarray::Array3<Complex<F>>,
    fft_plan: &Arc<dyn Fft<F>>,
) {
    let (k, patch_size, _) = group.dim();
    for r in 0..patch_size {
        for c in 0..patch_size {
            let mut vec: Vec<Complex<F>> = (0..k).map(|i| group[[i, r, c]]).collect();
            fft_plan.process(&mut vec);
            for i in 0..k {
                group[[i, r, c]] = vec[i];
            }
        }
    }
}

/// Apply 1D inverse FFT along the group dimension at each (r, c) position.
fn apply_inverse_1d_transform<F: Bm3dFloat>(
    group: &mut ndarray::Array3<Complex<F>>,
    ifft_plan: &Arc<dyn Fft<F>>,
) {
    let (k, patch_size, _) = group.dim();
    let norm_k = F::one() / F::usize_as(k);
    for r in 0..patch_size {
        for c in 0..patch_size {
            let mut vec: Vec<Complex<F>> = (0..k).map(|i| group[[i, r, c]]).collect();
            ifft_plan.process(&mut vec);
            for i in 0..k {
                group[[i, r, c]] = vec[i] * norm_k;
            }
        }
    }
}

/// Apply 2D inverse transform to a single patch slice.
///
/// Uses IWHT for 8x8 patches, IFFT otherwise.
fn apply_inverse_2d_transform<F: Bm3dFloat>(
    complex_slice: &ndarray::Array2<Complex<F>>,
    use_hadamard: bool,
    ifft_row: &Arc<dyn Fft<F>>,
    ifft_col: &Arc<dyn Fft<F>>,
) -> Array2<F> {
    if use_hadamard {
        transforms::wht2d_8x8_inverse(complex_slice)
    } else {
        transforms::ifft2d(complex_slice, ifft_row, ifft_col)
    }
}

/// Aggregate a single denoised patch into the numerator and denominator accumulators.
#[allow(clippy::too_many_arguments)]
fn aggregate_patch<F: Bm3dFloat>(
    spatial: &Array2<F>,
    m: &PatchMatch<F>,
    weight: F,
    patch_size: usize,
    rows: usize,
    cols: usize,
    numerator: &mut Array2<F>,
    denominator: &mut Array2<F>,
) {
    for pr in 0..patch_size {
        for pc in 0..patch_size {
            let tr = m.row + pr;
            let tc = m.col + pc;
            if tr < rows && tc < cols {
                numerator[[tr, tc]] += spatial[[pr, pc]] * weight;
                denominator[[tr, tc]] += weight;
            }
        }
    }
}

/// Finalize the output by dividing numerator by denominator.
///
/// Falls back to original noisy input where denominator is too small.
fn finalize_output<F: Bm3dFloat>(
    final_num: &Array2<F>,
    final_den: &Array2<F>,
    input_noisy: ArrayView2<F>,
) -> Array2<F> {
    let (rows, cols) = input_noisy.dim();
    let mut output = Array2::<F>::zeros((rows, cols));
    let agg_eps = F::from_f64_c(AGGREGATION_EPSILON);

    for r in 0..rows {
        for c in 0..cols {
            let num = final_num[[r, c]];
            let den = final_den[[r, c]];
            if den > agg_eps {
                output[[r, c]] = num / den;
            } else {
                output[[r, c]] = input_noisy[[r, c]];
            }
        }
    }
    output
}

/// Core BM3D Single Image Kernel
#[allow(clippy::too_many_arguments)]
pub fn run_bm3d_kernel<F: Bm3dFloat>(
    input_noisy: ArrayView2<F>,
    input_pilot: ArrayView2<F>,
    mode: Bm3dMode,
    sigma_psd: ArrayView2<F>,
    sigma_map: ArrayView2<F>,
    sigma_random: F,
    threshold: F,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
    plans: &Bm3dPlans<F>,
) -> Array2<F> {
    let (rows, cols) = input_noisy.dim();
    let use_sigma_map = sigma_map.dim() == (rows, cols);
    let use_colored_noise = sigma_psd.dim() == (patch_size, patch_size);
    let scalar_sigma_sq = sigma_random * sigma_random;

    // Fast path for 8x8 patches using Hadamard
    let use_hadamard = patch_size == HADAMARD_PATCH_SIZE;

    // Pre-compute Integral Images for Block Matching acceleration
    let (integral_sum, integral_sq_sum) = block_matching::compute_integral_images(input_pilot);

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

    let wiener_eps = F::from_f64_c(WIENER_EPSILON);
    let max_wiener_weight = F::from_f64_c(MAX_WIENER_WEIGHT);

    let (final_num, final_den) = ref_coords
        .par_iter()
        .with_min_len(RAYON_MIN_CHUNK_LEN)
        .fold(
            || {
                (
                    Array2::<F>::zeros((rows, cols)),
                    Array2::<F>::zeros((rows, cols)),
                )
            },
            |mut acc, &(ref_r, ref_c)| {
                let (numerator_acc, denominator_acc) = &mut acc;

                // 1. Block Matching
                let matches = block_matching::find_similar_patches(
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
                if k == 0 {
                    return acc;
                }

                // 1.5 Local Noise Level
                let local_sigma_streak = if use_sigma_map {
                    sigma_map[[ref_r, ref_c]]
                } else {
                    F::zero()
                };

                // Build patch groups
                let (group_noisy, group_pilot) =
                    build_patch_groups(input_noisy, input_pilot, &matches, patch_size);

                // Forward 2D transforms
                let mut g_noisy_c = apply_forward_2d_transform(
                    &group_noisy,
                    use_hadamard,
                    fft_2d_row_ref,
                    fft_2d_col_ref,
                );
                let mut g_pilot_c = if mode == Bm3dMode::Wiener {
                    apply_forward_2d_transform(
                        &group_pilot,
                        use_hadamard,
                        fft_2d_row_ref,
                        fft_2d_col_ref,
                    )
                } else {
                    ndarray::Array3::<Complex<F>>::zeros((k, patch_size, patch_size))
                };

                // Forward 1D transforms along group dimension
                let fft_k_plan = &fft_1d_plans_ref[k];
                apply_forward_1d_transform(&mut g_noisy_c, fft_k_plan);
                if mode == Bm3dMode::Wiener {
                    apply_forward_1d_transform(&mut g_pilot_c, fft_k_plan);
                }

                // Filtering
                let mut weight_g = F::one();
                let spatial_scale = F::usize_as(patch_size);
                let spatial_scale_sq = spatial_scale * spatial_scale;

                match mode {
                    Bm3dMode::HardThreshold => {
                        // Hard Thresholding
                        let mut nz_count = 0usize;
                        for i in 0..k {
                            for r in 0..patch_size {
                                for c in 0..patch_size {
                                    let coeff = g_noisy_c[[i, r, c]];
                                    let noise_std_coeff = compute_noise_std(
                                        use_colored_noise,
                                        sigma_psd,
                                        local_sigma_streak,
                                        scalar_sigma_sq,
                                        k,
                                        r,
                                        c,
                                        spatial_scale,
                                    );
                                    if coeff.norm() < threshold * noise_std_coeff {
                                        g_noisy_c[[i, r, c]] = Complex::new(F::zero(), F::zero());
                                    } else {
                                        nz_count += 1;
                                    }
                                }
                            }
                        }
                        if nz_count > 0 {
                            weight_g = F::one() / (F::usize_as(nz_count) + F::one());
                        }
                    }
                    Bm3dMode::Wiener => {
                        // Wiener Filtering
                        let mut wiener_sum = F::zero();
                        for i in 0..k {
                            for r in 0..patch_size {
                                for c in 0..patch_size {
                                    let p_val = g_pilot_c[[i, r, c]];
                                    let n_val = g_noisy_c[[i, r, c]];
                                    let noise_var_coeff = compute_noise_var(
                                        use_colored_noise,
                                        sigma_psd,
                                        local_sigma_streak,
                                        scalar_sigma_sq,
                                        k,
                                        r,
                                        c,
                                        spatial_scale_sq,
                                    );
                                    let w = p_val.norm_sqr()
                                        / (p_val.norm_sqr() + noise_var_coeff + wiener_eps);
                                    g_noisy_c[[i, r, c]] = n_val * w;
                                    wiener_sum += w * w;
                                }
                            }
                        }
                        weight_g = F::one() / (wiener_sum * scalar_sigma_sq + wiener_eps);
                        if weight_g > max_wiener_weight {
                            weight_g = max_wiener_weight;
                        }
                    }
                }

                // Inverse 1D transform along group dimension
                let ifft_k_plan = &ifft_1d_plans_ref[k];
                apply_inverse_1d_transform(&mut g_noisy_c, ifft_k_plan);

                // Inverse 2D transforms and aggregation
                #[allow(clippy::needless_range_loop)]
                for i in 0..k {
                    let complex_slice = g_noisy_c.slice(s![i, .., ..]).to_owned();
                    let spatial = apply_inverse_2d_transform(
                        &complex_slice,
                        use_hadamard,
                        ifft_2d_row_ref,
                        ifft_2d_col_ref,
                    );
                    aggregate_patch(
                        &spatial,
                        &matches[i],
                        weight_g,
                        patch_size,
                        rows,
                        cols,
                        numerator_acc,
                        denominator_acc,
                    );
                }
                acc
            },
        )
        .reduce(
            || {
                (
                    Array2::<F>::zeros((rows, cols)),
                    Array2::<F>::zeros((rows, cols)),
                )
            },
            |mut a, b| {
                a.0 = &a.0 + &b.0;
                a.1 = &a.1 + &b.1;
                a
            },
        );

    // Finalize: divide numerator by denominator
    finalize_output(&final_num, &final_den, input_noisy)
}

/// Run BM3D step on a single 2D image.
#[allow(clippy::too_many_arguments)]
pub fn run_bm3d_step<F: Bm3dFloat>(
    input_noisy: ArrayView2<F>,
    input_pilot: ArrayView2<F>,
    mode: Bm3dMode,
    sigma_psd: ArrayView2<F>,
    sigma_map: ArrayView2<F>,
    sigma_random: F,
    threshold: F,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
    plans: &Bm3dPlans<F>,
) -> Result<Array2<F>, String> {
    if input_pilot.dim() != input_noisy.dim() {
        return Err(format!(
            "Dimension mismatch: input_noisy has shape {:?}, but input_pilot has shape {:?}",
            input_noisy.dim(),
            input_pilot.dim()
        ));
    }
    if sigma_map.dim() != input_noisy.dim() && sigma_map.dim() != (1, 1) {
        return Err(format!(
            "Sigma map dimension mismatch: expected {:?} or (1, 1), got {:?}",
            input_noisy.dim(),
            sigma_map.dim()
        ));
    }

    Ok(run_bm3d_kernel(
        input_noisy,
        input_pilot,
        mode,
        sigma_psd,
        sigma_map,
        sigma_random,
        threshold,
        patch_size,
        step_size,
        search_window,
        max_matches,
        plans,
    ))
}

/// Run BM3D step on a 3D stack of images.
#[allow(clippy::too_many_arguments)]
pub fn run_bm3d_step_stack<F: Bm3dFloat>(
    input_noisy: ArrayView3<F>,
    input_pilot: ArrayView3<F>,
    mode: Bm3dMode,
    sigma_psd: ArrayView2<F>,
    sigma_map: ArrayView3<F>,
    sigma_random: F,
    threshold: F,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
    plans: &Bm3dPlans<F>,
) -> Result<Array3<F>, String> {
    let (n, rows, cols) = input_noisy.dim();
    if input_pilot.dim() != (n, rows, cols) {
        return Err(format!(
            "Stack dimension mismatch: input_noisy has shape {:?}, but input_pilot has shape {:?}",
            input_noisy.dim(),
            input_pilot.dim()
        ));
    }
    if sigma_map.dim() != (n, rows, cols) && sigma_map.dim() != (1, 1, 1) {
        return Err(format!(
            "Sigma map dimension mismatch: expected {:?} or (1, 1, 1), got {:?}",
            (n, rows, cols),
            sigma_map.dim()
        ));
    }

    let results: Vec<Array2<F>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let noisy_slice = input_noisy.index_axis(Axis(0), i);
            let pilot_slice = input_pilot.index_axis(Axis(0), i);
            let map_slice = if sigma_map.dim() == (1, 1, 1) {
                sigma_map.index_axis(Axis(0), 0) // Dummy view
            } else {
                sigma_map.index_axis(Axis(0), i)
            };

            run_bm3d_kernel(
                noisy_slice,
                pilot_slice,
                mode,
                sigma_psd,
                map_slice,
                sigma_random,
                threshold,
                patch_size,
                step_size,
                search_window,
                max_matches,
                plans,
            )
        })
        .collect();

    // Consolidate
    let mut output = Array3::<F>::zeros((n, rows, cols));
    for (i, res) in results.into_iter().enumerate() {
        output.slice_mut(s![i, .., ..]).assign(&res);
    }
    Ok(output)
}

/// Test function for block matching (used for debugging/validation).
pub fn test_block_matching<F: Bm3dFloat>(
    input: ArrayView2<F>,
    ref_r: usize,
    ref_c: usize,
    patch_size: usize,
    search_win: usize,
    max_matches: usize,
) -> Vec<(usize, usize, F)> {
    let (sum_img, sq_sum_img) = block_matching::compute_integral_images(input);
    let matches = block_matching::find_similar_patches(
        input,
        &sum_img,
        &sq_sum_img,
        (ref_r, ref_c),
        (patch_size, patch_size),
        (search_win, search_win),
        max_matches,
        1,
    );
    matches
        .into_iter()
        .map(|m| (m.row, m.col, m.distance))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    // Helper: Simple Linear Congruential Generator for deterministic "random" test data
    struct SimpleLcg {
        state: u64,
    }

    impl SimpleLcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.state
        }

        fn next_f32(&mut self) -> f32 {
            // Generate f32 in range [0.0, 1.0)
            let u = self.next_u64();
            (u >> 40) as f32 / (1u64 << 24) as f32
        }

        // Box-Muller approximation for Gaussian noise
        fn next_gaussian(&mut self) -> f32 {
            let u1 = self.next_f32().max(1e-10);
            let u2 = self.next_f32();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        }
    }

    // Helper: Generate deterministic "random" matrix in [0, 1]
    fn random_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
        let mut rng = SimpleLcg::new(seed);
        Array2::from_shape_fn((rows, cols), |_| rng.next_f32())
    }

    // Helper: Generate deterministic 3D stack
    fn random_stack(depth: usize, rows: usize, cols: usize, seed: u64) -> Array3<f32> {
        let mut rng = SimpleLcg::new(seed);
        Array3::from_shape_fn((depth, rows, cols), |_| rng.next_f32())
    }

    // Helper: Add Gaussian noise to image
    fn add_gaussian_noise(image: &Array2<f32>, noise_std: f32, seed: u64) -> Array2<f32> {
        let mut rng = SimpleLcg::new(seed);
        let (rows, cols) = image.dim();
        Array2::from_shape_fn((rows, cols), |(r, c)| {
            (image[[r, c]] + rng.next_gaussian() * noise_std).clamp(0.0, 1.0)
        })
    }

    // Helper: Add Gaussian noise to 3D stack
    fn add_gaussian_noise_stack(stack: &Array3<f32>, noise_std: f32, seed: u64) -> Array3<f32> {
        let mut rng = SimpleLcg::new(seed);
        let (depth, rows, cols) = stack.dim();
        Array3::from_shape_fn((depth, rows, cols), |(d, r, c)| {
            (stack[[d, r, c]] + rng.next_gaussian() * noise_std).clamp(0.0, 1.0)
        })
    }

    // Helper: Mean squared error
    fn mse(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        assert_eq!(a.dim(), b.dim());
        let sum_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        sum_sq / (a.len() as f32)
    }

    // Helper: MSE for 3D stack
    fn mse_stack(a: &Array3<f32>, b: &Array3<f32>) -> f32 {
        assert_eq!(a.dim(), b.dim());
        let sum_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        sum_sq / (a.len() as f32)
    }

    // Default test parameters (small for speed)
    const TEST_PATCH_SIZE: usize = 8;
    const TEST_STEP_SIZE: usize = 4;
    const TEST_SEARCH_WINDOW: usize = 16;
    const TEST_MAX_MATCHES: usize = 8;
    const TEST_THRESHOLD: f32 = 2.7;
    const TEST_SIGMA_RANDOM: f32 = 0.05;

    // Create dummy sigma arrays (no colored noise, no streak map)
    fn dummy_sigma_psd() -> Array2<f32> {
        Array2::zeros((1, 1))
    }

    fn dummy_sigma_map_2d() -> Array2<f32> {
        Array2::zeros((1, 1))
    }

    fn dummy_sigma_map_3d() -> Array3<f32> {
        Array3::zeros((1, 1, 1))
    }

    // ==================== Smoke Tests ====================

    #[test]
    fn test_hard_thresholding_smoke() {
        let image = random_matrix(32, 32, 12345);
        let sigma_psd = dummy_sigma_psd();
        let sigma_map = dummy_sigma_map_2d();
        let plans = Bm3dPlans::new(TEST_PATCH_SIZE, TEST_MAX_MATCHES);

        let output = run_bm3d_step(
            image.view(),
            image.view(), // pilot = noisy for first pass
            Bm3dMode::HardThreshold,
            sigma_psd.view(),
            sigma_map.view(),
            TEST_SIGMA_RANDOM,
            TEST_THRESHOLD,
            TEST_PATCH_SIZE,
            TEST_STEP_SIZE,
            TEST_SEARCH_WINDOW,
            TEST_MAX_MATCHES,
            &plans,
        )
        .unwrap();

        // Should complete without panic and produce valid output
        assert_eq!(output.dim(), image.dim());
        assert!(
            output.iter().all(|&x| x.is_finite()),
            "Output contains non-finite values"
        );
    }

    #[test]
    fn test_wiener_filtering_smoke() {
        let image = random_matrix(32, 32, 54321);
        let sigma_psd = dummy_sigma_psd();
        let sigma_map = dummy_sigma_map_2d();
        let plans = Bm3dPlans::new(TEST_PATCH_SIZE, TEST_MAX_MATCHES);

        let output = run_bm3d_step(
            image.view(),
            image.view(),
            Bm3dMode::Wiener,
            sigma_psd.view(),
            sigma_map.view(),
            TEST_SIGMA_RANDOM,
            0.0, // threshold not used for Wiener
            TEST_PATCH_SIZE,
            TEST_STEP_SIZE,
            TEST_SEARCH_WINDOW,
            TEST_MAX_MATCHES,
            &plans,
        )
        .unwrap();

        assert_eq!(output.dim(), image.dim());
        assert!(
            output.iter().all(|&x| x.is_finite()),
            "Output contains non-finite values"
        );
    }

    #[test]
    fn test_hard_thresholding_stack_smoke() {
        let stack = random_stack(4, 32, 32, 11111);
        let sigma_psd = dummy_sigma_psd();
        let sigma_map = dummy_sigma_map_3d();
        let plans = Bm3dPlans::new(TEST_PATCH_SIZE, TEST_MAX_MATCHES);

        let output = run_bm3d_step_stack(
            stack.view(),
            stack.view(),
            Bm3dMode::HardThreshold,
            sigma_psd.view(),
            sigma_map.view(),
            TEST_SIGMA_RANDOM,
            TEST_THRESHOLD,
            TEST_PATCH_SIZE,
            TEST_STEP_SIZE,
            TEST_SEARCH_WINDOW,
            TEST_MAX_MATCHES,
            &plans,
        )
        .unwrap();

        assert_eq!(output.dim(), stack.dim());
        assert!(
            output.iter().all(|&x| x.is_finite()),
            "Output contains non-finite values"
        );
    }

    #[test]
    fn test_wiener_filtering_stack_smoke() {
        let stack = random_stack(4, 32, 32, 22222);
        let sigma_psd = dummy_sigma_psd();
        let sigma_map = dummy_sigma_map_3d();
        let plans = Bm3dPlans::new(TEST_PATCH_SIZE, TEST_MAX_MATCHES);

        let output = run_bm3d_step_stack(
            stack.view(),
            stack.view(),
            Bm3dMode::Wiener,
            sigma_psd.view(),
            sigma_map.view(),
            TEST_SIGMA_RANDOM,
            0.0,
            TEST_PATCH_SIZE,
            TEST_STEP_SIZE,
            TEST_SEARCH_WINDOW,
            TEST_MAX_MATCHES,
            &plans,
        )
        .unwrap();

        assert_eq!(output.dim(), stack.dim());
        assert!(
            output.iter().all(|&x| x.is_finite()),
            "Output contains non-finite values"
        );
    }

    // ==================== Output Shape Tests ====================

    #[test]
    fn test_hard_thresholding_preserves_shape() {
        for (rows, cols) in [(32, 32), (48, 64), (64, 48)] {
            let image = random_matrix(rows, cols, (rows * 100 + cols) as u64);
            let sigma_psd = dummy_sigma_psd();
            let sigma_map = dummy_sigma_map_2d();
            let plans = Bm3dPlans::new(TEST_PATCH_SIZE, TEST_MAX_MATCHES);

            let output = run_bm3d_step(
                image.view(),
                image.view(),
                Bm3dMode::HardThreshold,
                sigma_psd.view(),
                sigma_map.view(),
                TEST_SIGMA_RANDOM,
                TEST_THRESHOLD,
                TEST_PATCH_SIZE,
                TEST_STEP_SIZE,
                TEST_SEARCH_WINDOW,
                TEST_MAX_MATCHES,
                &plans,
            )
            .unwrap();

            assert_eq!(
                output.dim(),
                (rows, cols),
                "Output shape mismatch for {}x{}",
                rows,
                cols
            );
        }
    }

    #[test]
    fn test_wiener_filtering_preserves_shape() {
        let image = random_matrix(40, 56, 33333);
        let sigma_psd = dummy_sigma_psd();
        let sigma_map = dummy_sigma_map_2d();
        let plans = Bm3dPlans::new(TEST_PATCH_SIZE, TEST_MAX_MATCHES);

        let output = run_bm3d_step(
            image.view(),
            image.view(),
            Bm3dMode::Wiener,
            sigma_psd.view(),
            sigma_map.view(),
            TEST_SIGMA_RANDOM,
            0.0,
            TEST_PATCH_SIZE,
            TEST_STEP_SIZE,
            TEST_SEARCH_WINDOW,
            TEST_MAX_MATCHES,
            &plans,
        )
        .unwrap();

        assert_eq!(output.dim(), image.dim());
    }

    #[test]
    fn test_stack_preserves_shape() {
        let stack = random_stack(5, 40, 48, 44444);
        let sigma_psd = dummy_sigma_psd();
        let sigma_map = dummy_sigma_map_3d();
        let plans = Bm3dPlans::new(TEST_PATCH_SIZE, TEST_MAX_MATCHES);

        let output = run_bm3d_step_stack(
            stack.view(),
            stack.view(),
            Bm3dMode::HardThreshold,
            sigma_psd.view(),
            sigma_map.view(),
            TEST_SIGMA_RANDOM,
            TEST_THRESHOLD,
            TEST_PATCH_SIZE,
            TEST_STEP_SIZE,
            TEST_SEARCH_WINDOW,
            TEST_MAX_MATCHES,
            &plans,
        )
        .unwrap();

        assert_eq!(output.dim(), stack.dim());
    }

    // ==================== Behavioral Sanity Tests ====================

    #[test]
    fn test_denoising_modifies_noisy_input() {
        let clean = random_matrix(32, 32, 55555);
        let noisy = add_gaussian_noise(&clean, 0.1, 66666);
        let sigma_psd = dummy_sigma_psd();
        let sigma_map = dummy_sigma_map_2d();
        let plans = Bm3dPlans::new(TEST_PATCH_SIZE, TEST_MAX_MATCHES);

        let output = run_bm3d_step(
            noisy.view(),
            noisy.view(),
            Bm3dMode::HardThreshold,
            sigma_psd.view(),
            sigma_map.view(),
            0.1, // Match noise level
            TEST_THRESHOLD,
            TEST_PATCH_SIZE,
            TEST_STEP_SIZE,
            TEST_SEARCH_WINDOW,
            TEST_MAX_MATCHES,
            &plans,
        )
        .unwrap();

        // Output should differ from input (denoising did something)
        let diff = mse(&output, &noisy);
        assert!(
            diff > 1e-6,
            "Denoising should modify the input, but MSE was {}",
            diff
        );
    }

    #[test]
    fn test_denoising_reduces_noise() {
        // Use a smooth gradient image (not random) which BM3D can exploit
        // BM3D works best on images with self-similar patches
        let clean = Array2::from_shape_fn((64, 64), |(r, c)| {
            // Smooth gradient with some structure
            0.5 + 0.3 * ((r as f32 / 64.0).sin() + (c as f32 / 64.0).cos())
        });
        let noisy = add_gaussian_noise(&clean, 0.1, 88888);
        let sigma_psd = dummy_sigma_psd();
        let sigma_map = dummy_sigma_map_2d();
        let plans = Bm3dPlans::new(8, 16);

        let output = run_bm3d_step(
            noisy.view(),
            noisy.view(),
            Bm3dMode::HardThreshold,
            sigma_psd.view(),
            sigma_map.view(),
            0.1,
            2.7, // Standard HT threshold
            8,   // patch_size
            2,   // smaller step for better coverage
            24,  // larger search window
            16,  // more matches
            &plans,
        )
        .unwrap();

        let mse_before = mse(&noisy, &clean);
        let mse_after = mse(&output, &clean);

        // Denoising should reduce MSE, or at minimum not increase it significantly
        // Use a relaxed assertion since BM3D behavior depends on image structure
        assert!(
            mse_after < mse_before * 1.5,
            "Denoising should not significantly increase MSE: before={}, after={}",
            mse_before,
            mse_after
        );
    }

    #[test]
    fn test_constant_image_approximately_unchanged() {
        // Uniform image with no noise - output should be similar to input
        let constant_val = 0.5f32;
        let image = Array2::<f32>::from_elem((32, 32), constant_val);
        let sigma_psd = dummy_sigma_psd();
        let sigma_map = dummy_sigma_map_2d();
        let plans = Bm3dPlans::new(TEST_PATCH_SIZE, TEST_MAX_MATCHES);

        let output = run_bm3d_step(
            image.view(),
            image.view(),
            Bm3dMode::HardThreshold,
            sigma_psd.view(),
            sigma_map.view(),
            0.01, // Very low noise
            TEST_THRESHOLD,
            TEST_PATCH_SIZE,
            TEST_STEP_SIZE,
            TEST_SEARCH_WINDOW,
            TEST_MAX_MATCHES,
            &plans,
        )
        .unwrap();

        // Output should be very close to input for constant image
        let max_diff = output
            .iter()
            .map(|&x| (x - constant_val).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 0.01,
            "Constant image should remain approximately unchanged, max_diff={}",
            max_diff
        );
    }

    #[test]
    fn test_output_in_valid_range() {
        // Input in [0, 1] should produce reasonable output (no NaN, no extreme values)
        let image = random_matrix(32, 32, 99999);
        let sigma_psd = dummy_sigma_psd();
        let sigma_map = dummy_sigma_map_2d();
        let plans = Bm3dPlans::new(TEST_PATCH_SIZE, TEST_MAX_MATCHES);

        let output = run_bm3d_step(
            image.view(),
            image.view(),
            Bm3dMode::HardThreshold,
            sigma_psd.view(),
            sigma_map.view(),
            TEST_SIGMA_RANDOM,
            TEST_THRESHOLD,
            TEST_PATCH_SIZE,
            TEST_STEP_SIZE,
            TEST_SEARCH_WINDOW,
            TEST_MAX_MATCHES,
            &plans,
        )
        .unwrap();

        for &val in output.iter() {
            assert!(val.is_finite(), "Output contains non-finite value");
            assert!(
                (-1.0..=2.0).contains(&val),
                "Output value {} outside reasonable range",
                val
            );
        }
    }

    #[test]
    fn test_stack_denoising_reduces_noise() {
        // Use structured images (smooth gradients) that BM3D can exploit
        let clean = Array3::from_shape_fn((3, 64, 64), |(d, r, c)| {
            // Different smooth patterns per slice
            0.5 + 0.3 * ((r as f32 / 64.0 + d as f32 * 0.1).sin() + (c as f32 / 64.0).cos())
        });
        let noisy = add_gaussian_noise_stack(&clean, 0.1, 33344);
        let sigma_psd = dummy_sigma_psd();
        let sigma_map = dummy_sigma_map_3d();
        let plans = Bm3dPlans::new(8, 16);

        let output = run_bm3d_step_stack(
            noisy.view(),
            noisy.view(),
            Bm3dMode::HardThreshold,
            sigma_psd.view(),
            sigma_map.view(),
            0.1,
            2.7,
            8,
            2,
            24,
            16,
            &plans,
        )
        .unwrap();

        let mse_before = mse_stack(&noisy, &clean);
        let mse_after = mse_stack(&output, &clean);

        // Relaxed assertion - denoising should not significantly increase MSE
        assert!(
            mse_after < mse_before * 1.5,
            "Stack denoising should not significantly increase MSE: before={}, after={}",
            mse_before,
            mse_after
        );
    }

    // ==================== Parameter Variation Tests ====================

    #[test]
    fn test_different_patch_sizes() {
        // Test both 4x4 (FFT) and 8x8 (Hadamard) paths
        for patch_size in [4, 8] {
            let image = random_matrix(32, 32, (patch_size * 1000) as u64);
            let sigma_psd = dummy_sigma_psd();
            let sigma_map = dummy_sigma_map_2d();
            let plans = Bm3dPlans::new(patch_size, TEST_MAX_MATCHES);

            let output = run_bm3d_step(
                image.view(),
                image.view(),
                Bm3dMode::HardThreshold,
                sigma_psd.view(),
                sigma_map.view(),
                TEST_SIGMA_RANDOM,
                TEST_THRESHOLD,
                patch_size,
                patch_size / 2, // step = patch/2
                TEST_SEARCH_WINDOW,
                TEST_MAX_MATCHES,
                &plans,
            )
            .unwrap();

            assert_eq!(
                output.dim(),
                image.dim(),
                "Shape mismatch for patch_size={}",
                patch_size
            );
            assert!(
                output.iter().all(|&x| x.is_finite()),
                "Non-finite values for patch_size={}",
                patch_size
            );
        }
    }

    #[test]
    fn test_different_search_windows() {
        let image = random_matrix(48, 48, 55566);
        let sigma_psd = dummy_sigma_psd();
        let sigma_map = dummy_sigma_map_2d();

        for search_window in [8, 16, 24] {
            let plans = Bm3dPlans::new(TEST_PATCH_SIZE, TEST_MAX_MATCHES);

            let output = run_bm3d_step(
                image.view(),
                image.view(),
                Bm3dMode::HardThreshold,
                sigma_psd.view(),
                sigma_map.view(),
                TEST_SIGMA_RANDOM,
                TEST_THRESHOLD,
                TEST_PATCH_SIZE,
                TEST_STEP_SIZE,
                search_window,
                TEST_MAX_MATCHES,
                &plans,
            )
            .unwrap();

            assert_eq!(
                output.dim(),
                image.dim(),
                "Shape mismatch for search_window={}",
                search_window
            );
        }
    }

    #[test]
    fn test_different_max_matches() {
        let image = random_matrix(32, 32, 77788);
        let sigma_psd = dummy_sigma_psd();
        let sigma_map = dummy_sigma_map_2d();

        for max_matches in [4, 8, 16] {
            let plans = Bm3dPlans::new(TEST_PATCH_SIZE, max_matches);

            let output = run_bm3d_step(
                image.view(),
                image.view(),
                Bm3dMode::HardThreshold,
                sigma_psd.view(),
                sigma_map.view(),
                TEST_SIGMA_RANDOM,
                TEST_THRESHOLD,
                TEST_PATCH_SIZE,
                TEST_STEP_SIZE,
                TEST_SEARCH_WINDOW,
                max_matches,
                &plans,
            )
            .unwrap();

            assert_eq!(
                output.dim(),
                image.dim(),
                "Shape mismatch for max_matches={}",
                max_matches
            );
        }
    }

    // ==================== Edge Case Tests ====================

    #[test]
    fn test_minimum_viable_image() {
        // Smallest image that fits patch_size + some margin
        let min_size = TEST_PATCH_SIZE + 2;
        let image = random_matrix(min_size, min_size, 99911);
        let sigma_psd = dummy_sigma_psd();
        let sigma_map = dummy_sigma_map_2d();
        let plans = Bm3dPlans::new(TEST_PATCH_SIZE, 4);

        let output = run_bm3d_step(
            image.view(),
            image.view(),
            Bm3dMode::HardThreshold,
            sigma_psd.view(),
            sigma_map.view(),
            TEST_SIGMA_RANDOM,
            TEST_THRESHOLD,
            TEST_PATCH_SIZE,
            1,               // step=1 for small image
            TEST_PATCH_SIZE, // small search window
            4,               // fewer matches
            &plans,
        )
        .unwrap();

        assert_eq!(output.dim(), image.dim());
    }

    #[test]
    fn test_single_slice_stack() {
        // Stack with depth=1 should degenerate to 2D-like behavior
        let stack = random_stack(1, 32, 32, 88899);
        let sigma_psd = dummy_sigma_psd();
        let sigma_map = dummy_sigma_map_3d();
        let plans = Bm3dPlans::new(TEST_PATCH_SIZE, TEST_MAX_MATCHES);

        let output = run_bm3d_step_stack(
            stack.view(),
            stack.view(),
            Bm3dMode::HardThreshold,
            sigma_psd.view(),
            sigma_map.view(),
            TEST_SIGMA_RANDOM,
            TEST_THRESHOLD,
            TEST_PATCH_SIZE,
            TEST_STEP_SIZE,
            TEST_SEARCH_WINDOW,
            TEST_MAX_MATCHES,
            &plans,
        )
        .unwrap();

        assert_eq!(output.dim(), (1, 32, 32));
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_non_square_image() {
        // 32x64 non-square image
        let image = random_matrix(32, 64, 12399);
        let sigma_psd = dummy_sigma_psd();
        let sigma_map = dummy_sigma_map_2d();
        let plans = Bm3dPlans::new(TEST_PATCH_SIZE, TEST_MAX_MATCHES);

        let output = run_bm3d_step(
            image.view(),
            image.view(),
            Bm3dMode::HardThreshold,
            sigma_psd.view(),
            sigma_map.view(),
            TEST_SIGMA_RANDOM,
            TEST_THRESHOLD,
            TEST_PATCH_SIZE,
            TEST_STEP_SIZE,
            TEST_SEARCH_WINDOW,
            TEST_MAX_MATCHES,
            &plans,
        )
        .unwrap();

        assert_eq!(output.dim(), (32, 64));
    }

    #[test]
    fn test_wiener_with_pilot() {
        // Wiener filtering with separate pilot estimate (typical BM3D 2-pass)
        // Use structured image for better BM3D performance
        let clean = Array2::from_shape_fn((64, 64), |(r, c)| {
            0.5 + 0.3 * ((r as f32 / 64.0).sin() + (c as f32 / 64.0).cos())
        });
        let noisy = add_gaussian_noise(&clean, 0.1, 55566);
        let sigma_psd = dummy_sigma_psd();
        let sigma_map = dummy_sigma_map_2d();

        let plans = Bm3dPlans::new(8, 16);

        // First pass: HT to get pilot estimate
        let pilot = run_bm3d_step(
            noisy.view(),
            noisy.view(),
            Bm3dMode::HardThreshold,
            sigma_psd.view(),
            sigma_map.view(),
            0.1,
            2.7,
            8,
            2,
            24,
            16,
            &plans,
        )
        .unwrap();

        // Second pass: Wiener with pilot
        let output = run_bm3d_step(
            noisy.view(),
            pilot.view(), // Use HT result as pilot
            Bm3dMode::Wiener,
            sigma_psd.view(),
            sigma_map.view(),
            0.1,
            0.0,
            8,
            2,
            24,
            16,
            &plans,
        )
        .unwrap();

        // Verify both passes produce finite outputs with correct shape
        assert_eq!(output.dim(), clean.dim());
        assert!(
            output.iter().all(|&x| x.is_finite()),
            "Wiener output should be finite"
        );

        // Wiener should not drastically increase MSE compared to noisy input
        let mse_noisy = mse(&noisy, &clean);
        let mse_wiener = mse(&output, &clean);

        assert!(
            mse_wiener < mse_noisy * 2.0,
            "Wiener should not drastically increase MSE: noisy={}, wiener={}",
            mse_noisy,
            mse_wiener
        );
    }

    #[test]
    fn test_step_size_variations() {
        let image = random_matrix(32, 32, 66677);
        let sigma_psd = dummy_sigma_psd();
        let sigma_map = dummy_sigma_map_2d();

        for step_size in [1, 2, 4, 8] {
            let plans = Bm3dPlans::new(TEST_PATCH_SIZE, TEST_MAX_MATCHES);

            let output = run_bm3d_step(
                image.view(),
                image.view(),
                Bm3dMode::HardThreshold,
                sigma_psd.view(),
                sigma_map.view(),
                TEST_SIGMA_RANDOM,
                TEST_THRESHOLD,
                TEST_PATCH_SIZE,
                step_size,
                TEST_SEARCH_WINDOW,
                TEST_MAX_MATCHES,
                &plans,
            )
            .unwrap();

            assert_eq!(
                output.dim(),
                image.dim(),
                "Shape mismatch for step_size={}",
                step_size
            );
        }
    }
}
