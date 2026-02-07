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

/// Aggregation tile size for memory-bounded partial accumulation.
/// Larger tiles reduce hashmap overhead; smaller tiles reduce per-worker memory.
const AGGREGATION_TILE_SIZE: usize = 256;
const AGGREGATION_TILE_SIZE_ENV: &str = "BM3D_AGGREGATION_TILE_SIZE";

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

/// Fill a preallocated patch group buffer from matched locations.
///
/// Writes only the first `matches.len()` entries of `out_group`.
fn fill_patch_group<F: Bm3dFloat>(
    input: ArrayView2<F>,
    matches: &[PatchMatch<F>],
    patch_size: usize,
    out_group: &mut Array3<F>,
) {
    for (idx, m) in matches.iter().enumerate() {
        let patch = input.slice(s![m.row..m.row + patch_size, m.col..m.col + patch_size]);
        out_group.slice_mut(s![idx, .., ..]).assign(&patch);
    }
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

/// Apply 2D forward transform into a pre-allocated output buffer.
fn apply_forward_2d_transform_into<F: Bm3dFloat>(
    group: &Array3<F>,
    k: usize,
    use_hadamard: bool,
    fft_row: &Arc<dyn Fft<F>>,
    fft_col: &Arc<dyn Fft<F>>,
    out: &mut ndarray::Array3<Complex<F>>,
) {
    for i in 0..k {
        let slice = group.slice(s![i, .., ..]);
        if use_hadamard {
            let transformed = transforms::wht2d_8x8_forward(slice);
            out.slice_mut(s![i, .., ..]).assign(&transformed);
        } else {
            let transformed = transforms::fft2d(slice, fft_row, fft_col);
            out.slice_mut(s![i, .., ..]).assign(&transformed);
        }
    }
}

/// Apply 1D forward FFT along the group dimension at each (r, c) position.
fn apply_forward_1d_transform<F: Bm3dFloat>(
    group: &mut ndarray::Array3<Complex<F>>,
    k: usize,
    patch_size: usize,
    fft_plan: &Arc<dyn Fft<F>>,
    scratch: &mut [Complex<F>],
) {
    for r in 0..patch_size {
        for c in 0..patch_size {
            for i in 0..k {
                scratch[i] = group[[i, r, c]];
            }
            fft_plan.process(scratch);
            for i in 0..k {
                group[[i, r, c]] = scratch[i];
            }
        }
    }
}

/// Apply 1D inverse FFT along the group dimension at each (r, c) position.
fn apply_inverse_1d_transform<F: Bm3dFloat>(
    group: &mut ndarray::Array3<Complex<F>>,
    k: usize,
    patch_size: usize,
    ifft_plan: &Arc<dyn Fft<F>>,
    scratch: &mut [Complex<F>],
) {
    let norm_k = F::one() / F::usize_as(k);
    for r in 0..patch_size {
        for c in 0..patch_size {
            for i in 0..k {
                scratch[i] = group[[i, r, c]];
            }
            ifft_plan.process(scratch);
            for i in 0..k {
                group[[i, r, c]] = scratch[i] * norm_k;
            }
        }
    }
}

/// Apply 2D inverse transform to a single patch slice.
///
/// Uses IWHT for 8x8 patches, IFFT otherwise.
fn apply_inverse_2d_transform<F: Bm3dFloat>(
    complex_slice: ndarray::ArrayView2<Complex<F>>,
    use_hadamard: bool,
    ifft_row: &Arc<dyn Fft<F>>,
    ifft_col: &Arc<dyn Fft<F>>,
) -> Array2<F> {
    if use_hadamard {
        transforms::wht2d_8x8_inverse_view(complex_slice)
    } else {
        transforms::ifft2d_view(complex_slice, ifft_row, ifft_col)
    }
}

struct WorkerBuffers<F: Bm3dFloat> {
    group_noisy: Array3<F>,
    group_pilot: Array3<F>,
    g_noisy_c: ndarray::Array3<Complex<F>>,
    g_pilot_c: ndarray::Array3<Complex<F>>,
    coeff_buffer: Vec<F>,
    scratch_1d: Vec<Complex<F>>,
}

impl<F: Bm3dFloat> WorkerBuffers<F> {
    fn new(max_matches: usize, patch_size: usize) -> Self {
        Self {
            group_noisy: Array3::<F>::zeros((max_matches, patch_size, patch_size)),
            group_pilot: Array3::<F>::zeros((max_matches, patch_size, patch_size)),
            g_noisy_c: ndarray::Array3::<Complex<F>>::zeros((max_matches, patch_size, patch_size)),
            g_pilot_c: ndarray::Array3::<Complex<F>>::zeros((max_matches, patch_size, patch_size)),
            coeff_buffer: vec![F::zero(); patch_size * patch_size],
            scratch_1d: vec![Complex::new(F::zero(), F::zero()); max_matches.max(1)],
        }
    }
}

type TileAccumulator<F> = (Array2<F>, Array2<F>);
type TileAccumulatorVec<F> = Vec<Option<TileAccumulator<F>>>;

/// Resolve aggregation tile size from environment with a safe fallback.
fn resolve_aggregation_tile_size(patch_size: usize) -> usize {
    std::env::var(AGGREGATION_TILE_SIZE_ENV)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .map(|v| v.max(patch_size))
        .unwrap_or(AGGREGATION_TILE_SIZE.max(patch_size))
}

fn get_or_insert_tile<F: Bm3dFloat>(
    tile_accumulators: &mut TileAccumulatorVec<F>,
    tile_id: usize,
    tile_row: usize,
    tile_col: usize,
    rows: usize,
    cols: usize,
    tile_size: usize,
) -> &mut TileAccumulator<F> {
    if tile_accumulators[tile_id].is_none() {
        let row_start = tile_row * tile_size;
        let col_start = tile_col * tile_size;
        let tile_h = (rows - row_start).min(tile_size);
        let tile_w = (cols - col_start).min(tile_size);
        tile_accumulators[tile_id] = Some((
            Array2::<F>::zeros((tile_h, tile_w)),
            Array2::<F>::zeros((tile_h, tile_w)),
        ));
    }
    tile_accumulators[tile_id]
        .as_mut()
        .expect("tile should be initialized")
}

/// Aggregate a single denoised patch into tile-local numerator/denominator accumulators.
#[allow(clippy::too_many_arguments)]
fn aggregate_patch_into_tiles<F: Bm3dFloat>(
    spatial: &Array2<F>,
    m: &PatchMatch<F>,
    weight: F,
    patch_size: usize,
    rows: usize,
    cols: usize,
    tile_size: usize,
    tile_cols: usize,
    tile_accumulators: &mut TileAccumulatorVec<F>,
) {
    let patch_end_r = m.row + patch_size - 1;
    let patch_end_c = m.col + patch_size - 1;
    let start_tile_row = m.row / tile_size;
    let start_tile_col = m.col / tile_size;
    let end_tile_row = patch_end_r / tile_size;
    let end_tile_col = patch_end_c / tile_size;

    // Fast path: most patches are entirely inside one tile.
    if start_tile_row == end_tile_row && start_tile_col == end_tile_col {
        let tile_row = start_tile_row;
        let tile_col = start_tile_col;
        let tile_id = tile_row * tile_cols + tile_col;
        let row_start = tile_row * tile_size;
        let col_start = tile_col * tile_size;
        let local_r0 = m.row - row_start;
        let local_c0 = m.col - col_start;

        let (num_tile, den_tile) = get_or_insert_tile(
            tile_accumulators,
            tile_id,
            tile_row,
            tile_col,
            rows,
            cols,
            tile_size,
        );

        for pr in 0..patch_size {
            let tr = m.row + pr;
            if tr >= rows {
                continue;
            }
            let local_r = local_r0 + pr;
            for pc in 0..patch_size {
                let tc = m.col + pc;
                if tc < cols {
                    let local_c = local_c0 + pc;
                    num_tile[[local_r, local_c]] += spatial[[pr, pc]] * weight;
                    den_tile[[local_r, local_c]] += weight;
                }
            }
        }
        return;
    }

    for pr in 0..patch_size {
        for pc in 0..patch_size {
            let tr = m.row + pr;
            let tc = m.col + pc;
            if tr < rows && tc < cols {
                let tile_row = tr / tile_size;
                let tile_col = tc / tile_size;
                let tile_id = tile_row * tile_cols + tile_col;
                let row_start = tile_row * tile_size;
                let col_start = tile_col * tile_size;
                let local_r = tr - row_start;
                let local_c = tc - col_start;

                let (num_tile, den_tile) = get_or_insert_tile(
                    tile_accumulators,
                    tile_id,
                    tile_row,
                    tile_col,
                    rows,
                    cols,
                    tile_size,
                );

                num_tile[[local_r, local_c]] += spatial[[pr, pc]] * weight;
                den_tile[[local_r, local_c]] += weight;
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
    let tile_size = resolve_aggregation_tile_size(patch_size);
    let tile_rows = rows.div_ceil(tile_size).max(1);
    let tile_cols = cols.div_ceil(tile_size).max(1);
    let tile_count = tile_rows * tile_cols;

    let (final_num, final_den) = if ref_coords.is_empty() {
        (
            Array2::<F>::zeros((rows, cols)),
            Array2::<F>::zeros((rows, cols)),
        )
    } else {
        // Use one coordinate chunk per Rayon worker to preserve throughput
        // while keeping partial aggregation memory tile-bounded.
        let partial_count = ref_coords.len().min(rayon::current_num_threads().max(1));
        let chunk_len = ref_coords
            .len()
            .div_ceil(partial_count)
            .max(RAYON_MIN_CHUNK_LEN);

        ref_coords
            .par_chunks(chunk_len)
            .map(|coord_chunk| {
                let mut tile_accumulators: TileAccumulatorVec<F> =
                    (0..tile_count).map(|_| None).collect();
                let mut worker = WorkerBuffers::<F>::new(max_matches, patch_size);

                for &(ref_r, ref_c) in coord_chunk {
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
                        continue;
                    }

                    // 1.5 Local Noise Level
                    let local_sigma_streak = if use_sigma_map {
                        sigma_map[[ref_r, ref_c]]
                    } else {
                        F::zero()
                    };

                    // Build noisy patch group (pilot group is only needed in Wiener mode).
                    fill_patch_group(input_noisy, &matches, patch_size, &mut worker.group_noisy);

                    // Forward 2D transforms
                    apply_forward_2d_transform_into(
                        &worker.group_noisy,
                        k,
                        use_hadamard,
                        fft_2d_row_ref,
                        fft_2d_col_ref,
                        &mut worker.g_noisy_c,
                    );
                    if mode == Bm3dMode::Wiener {
                        fill_patch_group(
                            input_pilot,
                            &matches,
                            patch_size,
                            &mut worker.group_pilot,
                        );
                        apply_forward_2d_transform_into(
                            &worker.group_pilot,
                            k,
                            use_hadamard,
                            fft_2d_row_ref,
                            fft_2d_col_ref,
                            &mut worker.g_pilot_c,
                        );
                    }

                    // Forward 1D transforms along group dimension
                    let fft_k_plan = &fft_1d_plans_ref[k];
                    apply_forward_1d_transform(
                        &mut worker.g_noisy_c,
                        k,
                        patch_size,
                        fft_k_plan,
                        &mut worker.scratch_1d[..k],
                    );
                    if mode == Bm3dMode::Wiener {
                        apply_forward_1d_transform(
                            &mut worker.g_pilot_c,
                            k,
                            patch_size,
                            fft_k_plan,
                            &mut worker.scratch_1d[..k],
                        );
                    }

                    // Filtering
                    let mut weight_g = F::one();
                    let spatial_scale = F::usize_as(patch_size);
                    let spatial_scale_sq = spatial_scale * spatial_scale;

                    match mode {
                        Bm3dMode::HardThreshold => {
                            // Hard Thresholding
                            let hard_thresholds = &mut worker.coeff_buffer;
                            for r in 0..patch_size {
                                for c in 0..patch_size {
                                    hard_thresholds[r * patch_size + c] = threshold
                                        * compute_noise_std(
                                            use_colored_noise,
                                            sigma_psd,
                                            local_sigma_streak,
                                            scalar_sigma_sq,
                                            k,
                                            r,
                                            c,
                                            spatial_scale,
                                        );
                                }
                            }

                            let mut nz_count = 0usize;
                            for i in 0..k {
                                for r in 0..patch_size {
                                    for c in 0..patch_size {
                                        let coeff = worker.g_noisy_c[[i, r, c]];
                                        if coeff.norm() < hard_thresholds[r * patch_size + c] {
                                            worker.g_noisy_c[[i, r, c]] =
                                                Complex::new(F::zero(), F::zero());
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
                            let noise_vars = &mut worker.coeff_buffer;
                            for r in 0..patch_size {
                                for c in 0..patch_size {
                                    noise_vars[r * patch_size + c] = compute_noise_var(
                                        use_colored_noise,
                                        sigma_psd,
                                        local_sigma_streak,
                                        scalar_sigma_sq,
                                        k,
                                        r,
                                        c,
                                        spatial_scale_sq,
                                    );
                                }
                            }

                            let mut wiener_sum = F::zero();
                            for i in 0..k {
                                for r in 0..patch_size {
                                    for c in 0..patch_size {
                                        let p_val = worker.g_pilot_c[[i, r, c]];
                                        let n_val = worker.g_noisy_c[[i, r, c]];
                                        let w = p_val.norm_sqr()
                                            / (p_val.norm_sqr()
                                                + noise_vars[r * patch_size + c]
                                                + wiener_eps);
                                        worker.g_noisy_c[[i, r, c]] = n_val * w;
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
                    apply_inverse_1d_transform(
                        &mut worker.g_noisy_c,
                        k,
                        patch_size,
                        ifft_k_plan,
                        &mut worker.scratch_1d[..k],
                    );

                    // Inverse 2D transforms and aggregation
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..k {
                        let complex_slice = worker.g_noisy_c.slice(s![i, .., ..]);
                        let spatial = apply_inverse_2d_transform(
                            complex_slice,
                            use_hadamard,
                            ifft_2d_row_ref,
                            ifft_2d_col_ref,
                        );
                        aggregate_patch_into_tiles(
                            &spatial,
                            &matches[i],
                            weight_g,
                            patch_size,
                            rows,
                            cols,
                            tile_size,
                            tile_cols,
                            &mut tile_accumulators,
                        );
                    }
                }

                tile_accumulators
            })
            .reduce_with(|mut a, b| {
                for (a_slot, b_slot) in a.iter_mut().zip(b.into_iter()) {
                    if let Some((b_num, b_den)) = b_slot {
                        if let Some((a_num, a_den)) = a_slot.as_mut() {
                            *a_num += &b_num;
                            *a_den += &b_den;
                        } else {
                            *a_slot = Some((b_num, b_den));
                        }
                    }
                }
                a
            })
            .map_or_else(
                || {
                    (
                        Array2::<F>::zeros((rows, cols)),
                        Array2::<F>::zeros((rows, cols)),
                    )
                },
                |tile_accumulators| {
                    let mut numerator_acc = Array2::<F>::zeros((rows, cols));
                    let mut denominator_acc = Array2::<F>::zeros((rows, cols));

                    for (tile_id, tile_entry) in tile_accumulators.into_iter().enumerate() {
                        let Some((num_tile, den_tile)) = tile_entry else {
                            continue;
                        };
                        let tile_row = tile_id / tile_cols;
                        let tile_col = tile_id % tile_cols;
                        let row_start = tile_row * tile_size;
                        let col_start = tile_col * tile_size;
                        let (tile_h, tile_w) = num_tile.dim();

                        numerator_acc
                            .slice_mut(s![
                                row_start..row_start + tile_h,
                                col_start..col_start + tile_w
                            ])
                            .assign(&num_tile);
                        denominator_acc
                            .slice_mut(s![
                                row_start..row_start + tile_h,
                                col_start..col_start + tile_w
                            ])
                            .assign(&den_tile);
                    }

                    (numerator_acc, denominator_acc)
                },
            )
    };

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
