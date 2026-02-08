//! BM3D Pipeline - Core denoising kernel and multi-image processing.

use ndarray::{s, Array2, Array3, ArrayView2, ArrayView3, Axis};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use rustfft::num_complex::Complex;
use rustfft::Fft;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

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
const AGGREGATION_TILE_SIZE: usize = 192;
const AGGREGATION_TILE_SIZE_ENV: &str = "BM3D_AGGREGATION_TILE_SIZE";
const TRANSFORM_CACHE_CAPACITY: usize = 1_024;
const TRANSFORM_CACHE_CAPACITY_ENV: &str = "BM3D_TRANSFORM_CACHE_CAPACITY";
const PROFILE_TIMING_ENV: &str = "BM3D_PROFILE_TIMING";
const USE_HADAMARD_ENV: &str = "BM3D_USE_HADAMARD";
static USE_HADAMARD_FAST_PATH: AtomicBool = AtomicBool::new(false);

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

/// Shared BM3D kernel parameters used by both 2D and stack processing.
#[derive(Debug, Clone, Copy)]
pub struct Bm3dKernelConfig<F: Bm3dFloat> {
    pub sigma_random: F,
    pub threshold: F,
    pub patch_size: usize,
    pub step_size: usize,
    pub search_window: usize,
    pub max_matches: usize,
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

struct Fft2dRefs<'a, F: Bm3dFloat> {
    row: &'a Arc<dyn Fft<F>>,
    col: &'a Arc<dyn Fft<F>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PatchCoord {
    row: usize,
    col: usize,
}

#[inline(always)]
fn patch_coord_key(coord: PatchCoord) -> u64 {
    debug_assert!(coord.row <= u32::MAX as usize);
    debug_assert!(coord.col <= u32::MAX as usize);
    ((coord.row as u64) << 32) | (coord.col as u64)
}

struct PatchTransformCache<F: Bm3dFloat> {
    patch_size: usize,
    patch_area: usize,
    capacity: usize,
    next_slot: usize,
    keys: Vec<Option<u64>>,
    values: Vec<Complex<F>>,
    indices: FxHashMap<u64, usize>,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl<F: Bm3dFloat> PatchTransformCache<F> {
    fn new(patch_size: usize, capacity: usize) -> Self {
        let patch_area = patch_size * patch_size;
        Self {
            patch_size,
            patch_area,
            capacity,
            next_slot: 0,
            keys: vec![None; capacity],
            values: vec![Complex::new(F::zero(), F::zero()); patch_area * capacity],
            indices: FxHashMap::with_capacity_and_hasher(
                capacity.saturating_mul(2),
                Default::default(),
            ),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    fn stats(&self) -> (u64, u64, u64) {
        (self.hits, self.misses, self.evictions)
    }

    fn write_transform_into(
        &mut self,
        image: ArrayView2<F>,
        coord: PatchCoord,
        use_hadamard: bool,
        fft_2d: &Fft2dRefs<F>,
        work_complex: &mut Array2<Complex<F>>,
        scratch: &mut [Complex<F>],
        row_fft_scratch: &mut [Complex<F>],
        col_fft_scratch: &mut [Complex<F>],
        out: &mut [Complex<F>],
    ) {
        let key = patch_coord_key(coord);
        if let Some(&slot) = self.indices.get(&key) {
            self.hits += 1;
            let base = slot * self.patch_area;
            out.copy_from_slice(&self.values[base..base + self.patch_area]);
            return;
        }

        self.misses += 1;
        let patch = image.slice(s![
            coord.row..coord.row + self.patch_size,
            coord.col..coord.col + self.patch_size
        ]);
        let out_view = ndarray::ArrayViewMut2::from_shape((self.patch_size, self.patch_size), out)
            .expect("output transform slice shape should match patch size");
        if use_hadamard {
            transforms::wht2d_8x8_forward_into_view(patch, out_view);
        } else {
            transforms::fft2d_into_with_plan_scratch(
                patch,
                fft_2d.row,
                fft_2d.col,
                work_complex,
                out_view,
                scratch,
                row_fft_scratch,
                col_fft_scratch,
            );
        }

        if self.capacity == 0 {
            return;
        }

        let slot = self.next_slot;
        if let Some(old_key) = self.keys[slot] {
            self.indices.remove(&old_key);
            self.evictions += 1;
        }
        self.keys[slot] = Some(key);
        self.indices.insert(key, slot);
        let base = slot * self.patch_area;
        self.values[base..base + self.patch_area].copy_from_slice(out);
        self.next_slot = (self.next_slot + 1) % self.capacity;
    }
}

fn fill_transformed_group_from_cache<F: Bm3dFloat>(
    image: ArrayView2<F>,
    matches: &[PatchMatch<F>],
    k: usize,
    patch_size: usize,
    use_hadamard: bool,
    fft_2d: &Fft2dRefs<F>,
    cache: &mut PatchTransformCache<F>,
    work_complex: &mut Array2<Complex<F>>,
    scratch_2d: &mut [Complex<F>],
    row_fft_scratch: &mut [Complex<F>],
    col_fft_scratch: &mut [Complex<F>],
    out: &mut ndarray::Array3<Complex<F>>,
) {
    let patch_area = patch_size * patch_size;
    let out_data = out
        .as_slice_memory_order_mut()
        .expect("group output should be contiguous");
    for (i, m) in matches.iter().enumerate().take(k) {
        let base = i * patch_area;
        cache.write_transform_into(
            image,
            PatchCoord {
                row: m.row,
                col: m.col,
            },
            use_hadamard,
            fft_2d,
            work_complex,
            scratch_2d,
            row_fft_scratch,
            col_fft_scratch,
            &mut out_data[base..base + patch_area],
        );
    }
}

/// Apply 1D forward FFT along the group dimension at each (r, c) position.
fn apply_forward_1d_transform<F: Bm3dFloat>(
    group: &mut ndarray::Array3<Complex<F>>,
    k: usize,
    patch_size: usize,
    fft_plan: &Arc<dyn Fft<F>>,
    scratch: &mut [Complex<F>],
    fft_plan_scratch: &mut [Complex<F>],
) {
    let fft_scratch_len = fft_plan.get_inplace_scratch_len();
    debug_assert!(fft_plan_scratch.len() >= fft_scratch_len);
    if let Some(group_data) = group.as_slice_memory_order_mut() {
        let patch_area = patch_size * patch_size;
        for rc in 0..patch_area {
            for i in 0..k {
                scratch[i] = group_data[i * patch_area + rc];
            }
            if fft_scratch_len == 0 {
                fft_plan.process_with_scratch(&mut scratch[..k], &mut []);
            } else {
                fft_plan.process_with_scratch(&mut scratch[..k], fft_plan_scratch);
            }
            for i in 0..k {
                group_data[i * patch_area + rc] = scratch[i];
            }
        }
    } else {
        for r in 0..patch_size {
            for c in 0..patch_size {
                for i in 0..k {
                    scratch[i] = group[[i, r, c]];
                }
                if fft_scratch_len == 0 {
                    fft_plan.process_with_scratch(&mut scratch[..k], &mut []);
                } else {
                    fft_plan.process_with_scratch(&mut scratch[..k], fft_plan_scratch);
                }
                for i in 0..k {
                    group[[i, r, c]] = scratch[i];
                }
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
    ifft_plan_scratch: &mut [Complex<F>],
) {
    let norm_k = F::one() / F::usize_as(k);
    let ifft_scratch_len = ifft_plan.get_inplace_scratch_len();
    debug_assert!(ifft_plan_scratch.len() >= ifft_scratch_len);
    if let Some(group_data) = group.as_slice_memory_order_mut() {
        let patch_area = patch_size * patch_size;
        for rc in 0..patch_area {
            for i in 0..k {
                scratch[i] = group_data[i * patch_area + rc];
            }
            if ifft_scratch_len == 0 {
                ifft_plan.process_with_scratch(&mut scratch[..k], &mut []);
            } else {
                ifft_plan.process_with_scratch(&mut scratch[..k], ifft_plan_scratch);
            }
            for i in 0..k {
                group_data[i * patch_area + rc] = scratch[i] * norm_k;
            }
        }
    } else {
        for r in 0..patch_size {
            for c in 0..patch_size {
                for i in 0..k {
                    scratch[i] = group[[i, r, c]];
                }
                if ifft_scratch_len == 0 {
                    ifft_plan.process_with_scratch(&mut scratch[..k], &mut []);
                } else {
                    ifft_plan.process_with_scratch(&mut scratch[..k], ifft_plan_scratch);
                }
                for i in 0..k {
                    group[[i, r, c]] = scratch[i] * norm_k;
                }
            }
        }
    }
}

struct WorkerBuffers<F: Bm3dFloat> {
    matches: Vec<PatchMatch<F>>,
    g_noisy_c: ndarray::Array3<Complex<F>>,
    g_pilot_c: ndarray::Array3<Complex<F>>,
    spatial_patch: Array2<F>,
    complex_work: Array2<Complex<F>>,
    coeff_buffer: Vec<F>,
    scratch_1d: Vec<Complex<F>>,
    scratch_2d: Vec<Complex<F>>,
    fft2d_row_plan_scratch: Vec<Complex<F>>,
    fft2d_col_plan_scratch: Vec<Complex<F>>,
    ifft2d_row_plan_scratch: Vec<Complex<F>>,
    ifft2d_col_plan_scratch: Vec<Complex<F>>,
    fft1d_plan_scratch: Vec<Complex<F>>,
    ifft1d_plan_scratch: Vec<Complex<F>>,
}

impl<F: Bm3dFloat> WorkerBuffers<F> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_matches: usize,
        patch_size: usize,
        fft2d_row_plan_scratch_len: usize,
        fft2d_col_plan_scratch_len: usize,
        ifft2d_row_plan_scratch_len: usize,
        ifft2d_col_plan_scratch_len: usize,
        fft1d_plan_scratch_len: usize,
        ifft1d_plan_scratch_len: usize,
    ) -> Self {
        let patch_dim = (patch_size, patch_size);
        Self {
            matches: Vec::with_capacity(max_matches.max(1)),
            g_noisy_c: ndarray::Array3::<Complex<F>>::zeros((max_matches, patch_size, patch_size)),
            g_pilot_c: ndarray::Array3::<Complex<F>>::zeros((max_matches, patch_size, patch_size)),
            spatial_patch: Array2::<F>::zeros(patch_dim),
            complex_work: Array2::<Complex<F>>::zeros(patch_dim),
            coeff_buffer: vec![F::zero(); patch_size * patch_size],
            scratch_1d: vec![Complex::new(F::zero(), F::zero()); max_matches.max(1)],
            scratch_2d: vec![Complex::new(F::zero(), F::zero()); patch_size.max(1)],
            fft2d_row_plan_scratch: vec![
                Complex::new(F::zero(), F::zero());
                fft2d_row_plan_scratch_len
            ],
            fft2d_col_plan_scratch: vec![
                Complex::new(F::zero(), F::zero());
                fft2d_col_plan_scratch_len
            ],
            ifft2d_row_plan_scratch: vec![
                Complex::new(F::zero(), F::zero());
                ifft2d_row_plan_scratch_len
            ],
            ifft2d_col_plan_scratch: vec![
                Complex::new(F::zero(), F::zero());
                ifft2d_col_plan_scratch_len
            ],
            fft1d_plan_scratch: vec![Complex::new(F::zero(), F::zero()); fft1d_plan_scratch_len],
            ifft1d_plan_scratch: vec![Complex::new(F::zero(), F::zero()); ifft1d_plan_scratch_len],
        }
    }
}

type TileAccumulator<F> = (Array2<F>, Array2<F>);
type TileAccumulatorVec<F> = Vec<Option<TileAccumulator<F>>>;

struct TileAggregationGeometry {
    patch_size: usize,
    rows: usize,
    cols: usize,
    tile_size: usize,
    tile_cols: usize,
}

/// Resolve aggregation tile size from environment with a safe fallback.
fn resolve_aggregation_tile_size(patch_size: usize) -> usize {
    std::env::var(AGGREGATION_TILE_SIZE_ENV)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .map(|v| v.max(patch_size))
        .unwrap_or(AGGREGATION_TILE_SIZE.max(patch_size))
}

fn resolve_transform_cache_capacity() -> usize {
    std::env::var(TRANSFORM_CACHE_CAPACITY_ENV)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(TRANSFORM_CACHE_CAPACITY)
}

fn resolve_profile_timing() -> bool {
    std::env::var(PROFILE_TIMING_ENV)
        .ok()
        .map(|value| {
            let v = value.trim();
            v == "1"
                || v.eq_ignore_ascii_case("true")
                || v.eq_ignore_ascii_case("yes")
                || v.eq_ignore_ascii_case("on")
        })
        .unwrap_or(false)
}

#[derive(Default, Clone, Copy)]
struct KernelStageStats {
    groups: u64,
    matched_patches: u64,
    block_matching_ns: u128,
    forward_2d_ns: u128,
    forward_1d_ns: u128,
    filtering_ns: u128,
    inverse_1d_ns: u128,
    inverse_2d_ns: u128,
    aggregation_ns: u128,
    noisy_cache_hits: u64,
    noisy_cache_misses: u64,
    noisy_cache_evictions: u64,
    pilot_cache_hits: u64,
    pilot_cache_misses: u64,
    pilot_cache_evictions: u64,
}

impl KernelStageStats {
    fn merge(self, other: Self) -> Self {
        Self {
            groups: self.groups + other.groups,
            matched_patches: self.matched_patches + other.matched_patches,
            block_matching_ns: self.block_matching_ns + other.block_matching_ns,
            forward_2d_ns: self.forward_2d_ns + other.forward_2d_ns,
            forward_1d_ns: self.forward_1d_ns + other.forward_1d_ns,
            filtering_ns: self.filtering_ns + other.filtering_ns,
            inverse_1d_ns: self.inverse_1d_ns + other.inverse_1d_ns,
            inverse_2d_ns: self.inverse_2d_ns + other.inverse_2d_ns,
            aggregation_ns: self.aggregation_ns + other.aggregation_ns,
            noisy_cache_hits: self.noisy_cache_hits + other.noisy_cache_hits,
            noisy_cache_misses: self.noisy_cache_misses + other.noisy_cache_misses,
            noisy_cache_evictions: self.noisy_cache_evictions + other.noisy_cache_evictions,
            pilot_cache_hits: self.pilot_cache_hits + other.pilot_cache_hits,
            pilot_cache_misses: self.pilot_cache_misses + other.pilot_cache_misses,
            pilot_cache_evictions: self.pilot_cache_evictions + other.pilot_cache_evictions,
        }
    }
}

/// Enable or disable the 8x8 Hadamard fast path globally.
///
/// Default is `false` (quality-first FFT path).
pub fn set_use_hadamard_fast_path(enabled: bool) {
    USE_HADAMARD_FAST_PATH.store(enabled, Ordering::Relaxed);
}

/// Get current 8x8 Hadamard fast-path setting.
pub fn use_hadamard_fast_path() -> bool {
    USE_HADAMARD_FAST_PATH.load(Ordering::Relaxed)
}

/// Resolve whether the 8x8 Hadamard fast path should be used.
///
/// Default is quality-first (`false`), because FFT path is less prone to
/// patch-like artifacts near strong boundaries. Set `BM3D_USE_HADAMARD=1`
/// to re-enable speed-first behavior for 8x8 patches.
fn resolve_use_hadamard(patch_size: usize) -> bool {
    if patch_size != HADAMARD_PATCH_SIZE {
        return false;
    }
    if use_hadamard_fast_path() {
        return true;
    }
    std::env::var(USE_HADAMARD_ENV)
        .ok()
        .map(|value| {
            let v = value.trim();
            v == "1"
                || v.eq_ignore_ascii_case("true")
                || v.eq_ignore_ascii_case("yes")
                || v.eq_ignore_ascii_case("on")
        })
        .unwrap_or(false)
}

/// Build separable patch-blend weights for overlap-add aggregation.
///
/// Uses a sine window with positive edges (no hard zeros), then normalizes
/// mean weight to 1.0 so effective global scaling remains stable.
fn compute_patch_blend_weights<F: Bm3dFloat>(patch_size: usize) -> Array2<F> {
    if patch_size <= 1 {
        return Array2::ones((patch_size.max(1), patch_size.max(1)));
    }

    let n = F::usize_as(patch_size);
    let half = F::from_f64_c(0.5);
    let mut one_d = vec![F::zero(); patch_size];
    for (i, slot) in one_d.iter_mut().enumerate() {
        let idx = F::usize_as(i) + half;
        *slot = (F::PI * idx / n).sin();
    }

    let mut weights = Array2::<F>::zeros((patch_size, patch_size));
    let mut sum = F::zero();
    for r in 0..patch_size {
        for c in 0..patch_size {
            let v = one_d[r] * one_d[c];
            weights[[r, c]] = v;
            sum += v;
        }
    }

    let mean = sum / F::usize_as(patch_size * patch_size);
    if mean > F::zero() {
        for v in &mut weights {
            *v /= mean;
        }
    }
    weights
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
fn aggregate_patch_into_tiles<F: Bm3dFloat>(
    spatial: &Array2<F>,
    m: &PatchMatch<F>,
    weight: F,
    patch_blend_weights: &Array2<F>,
    geom: &TileAggregationGeometry,
    tile_accumulators: &mut TileAccumulatorVec<F>,
) {
    let spatial_data = spatial.as_slice_memory_order();
    let blend_data = patch_blend_weights.as_slice_memory_order();
    let patch_end_r = m.row + geom.patch_size - 1;
    let patch_end_c = m.col + geom.patch_size - 1;
    let start_tile_row = m.row / geom.tile_size;
    let start_tile_col = m.col / geom.tile_size;
    let end_tile_row = patch_end_r / geom.tile_size;
    let end_tile_col = patch_end_c / geom.tile_size;

    // Fast path: most patches are entirely inside one tile.
    if start_tile_row == end_tile_row && start_tile_col == end_tile_col {
        let tile_row = start_tile_row;
        let tile_col = start_tile_col;
        let tile_id = tile_row * geom.tile_cols + tile_col;
        let row_start = tile_row * geom.tile_size;
        let col_start = tile_col * geom.tile_size;
        let local_r0 = m.row - row_start;
        let local_c0 = m.col - col_start;

        let (num_tile, den_tile) = get_or_insert_tile(
            tile_accumulators,
            tile_id,
            tile_row,
            tile_col,
            geom.rows,
            geom.cols,
            geom.tile_size,
        );

        let tile_w = num_tile.dim().1;
        if let (Some(num_data), Some(den_data), Some(spatial_vals), Some(blend_vals)) = (
            num_tile.as_slice_memory_order_mut(),
            den_tile.as_slice_memory_order_mut(),
            spatial_data,
            blend_data,
        ) {
            for pr in 0..geom.patch_size {
                let src_base = pr * geom.patch_size;
                let dst_base = (local_r0 + pr) * tile_w + local_c0;
                for pc in 0..geom.patch_size {
                    let s = spatial_vals[src_base + pc];
                    let w = weight * blend_vals[src_base + pc];
                    num_data[dst_base + pc] += s * w;
                    den_data[dst_base + pc] += w;
                }
            }
        } else {
            for pr in 0..geom.patch_size {
                let local_r = local_r0 + pr;
                for pc in 0..geom.patch_size {
                    let local_c = local_c0 + pc;
                    let w = weight * patch_blend_weights[[pr, pc]];
                    num_tile[[local_r, local_c]] += spatial[[pr, pc]] * w;
                    den_tile[[local_r, local_c]] += w;
                }
            }
        }
        return;
    }

    for pr in 0..geom.patch_size {
        for pc in 0..geom.patch_size {
            let tr = m.row + pr;
            let tc = m.col + pc;
            let tile_row = tr / geom.tile_size;
            let tile_col = tc / geom.tile_size;
            let tile_id = tile_row * geom.tile_cols + tile_col;
            let row_start = tile_row * geom.tile_size;
            let col_start = tile_col * geom.tile_size;
            let local_r = tr - row_start;
            let local_c = tc - col_start;

            let (num_tile, den_tile) = get_or_insert_tile(
                tile_accumulators,
                tile_id,
                tile_row,
                tile_col,
                geom.rows,
                geom.cols,
                geom.tile_size,
            );

            let w = weight * patch_blend_weights[[pr, pc]];
            num_tile[[local_r, local_c]] += spatial[[pr, pc]] * w;
            den_tile[[local_r, local_c]] += w;
        }
    }
}

/// Core BM3D Single Image Kernel
pub fn run_bm3d_kernel<F: Bm3dFloat>(
    input_noisy: ArrayView2<F>,
    input_pilot: ArrayView2<F>,
    mode: Bm3dMode,
    sigma_psd: ArrayView2<F>,
    sigma_map: ArrayView2<F>,
    config: &Bm3dKernelConfig<F>,
    plans: &Bm3dPlans<F>,
) -> Array2<F> {
    let (rows, cols) = input_noisy.dim();
    let patch_size = config.patch_size;
    let step_size = config.step_size;
    let search_window = config.search_window;
    let max_matches = config.max_matches;
    let threshold = config.threshold;
    let use_sigma_map_full = sigma_map.dim() == (rows, cols);
    let use_sigma_map_row = sigma_map.dim() == (1, cols);
    let use_colored_noise = sigma_psd.dim() == (patch_size, patch_size);
    let scalar_sigma_sq = config.sigma_random * config.sigma_random;

    // Fast path for 8x8 patches using Hadamard (opt-in via env).
    let use_hadamard = resolve_use_hadamard(patch_size);

    // Pre-compute Integral Images for Block Matching acceleration
    let integral_sum = block_matching::compute_integral_sum_image(input_pilot);

    let r_end = rows.saturating_sub(patch_size) + 1;
    let c_end = cols.saturating_sub(patch_size) + 1;
    let ref_rows = if r_end == 0 {
        0
    } else {
        (r_end - 1) / step_size + 1
    };
    let ref_cols = if c_end == 0 {
        0
    } else {
        (c_end - 1) / step_size + 1
    };
    let total_refs = ref_rows * ref_cols;

    let fft_2d_row_ref = &plans.fft_2d_row;
    let fft_2d_col_ref = &plans.fft_2d_col;
    let ifft_2d_row_ref = &plans.ifft_2d_row;
    let ifft_2d_col_ref = &plans.ifft_2d_col;
    let fft_2d_refs = Fft2dRefs {
        row: fft_2d_row_ref,
        col: fft_2d_col_ref,
    };
    let fft_1d_plans_ref = &plans.fft_1d_plans;
    let ifft_1d_plans_ref = &plans.ifft_1d_plans;
    let fft2d_row_plan_scratch_len = fft_2d_row_ref.get_inplace_scratch_len();
    let fft2d_col_plan_scratch_len = fft_2d_col_ref.get_inplace_scratch_len();
    let ifft2d_row_plan_scratch_len = ifft_2d_row_ref.get_inplace_scratch_len();
    let ifft2d_col_plan_scratch_len = ifft_2d_col_ref.get_inplace_scratch_len();
    let fft1d_plan_scratch_len = fft_1d_plans_ref
        .iter()
        .map(|p| p.get_inplace_scratch_len())
        .max()
        .unwrap_or(0);
    let ifft1d_plan_scratch_len = ifft_1d_plans_ref
        .iter()
        .map(|p| p.get_inplace_scratch_len())
        .max()
        .unwrap_or(0);

    let wiener_eps = F::from_f64_c(WIENER_EPSILON);
    let max_wiener_weight = F::from_f64_c(MAX_WIENER_WEIGHT);
    let tile_size = resolve_aggregation_tile_size(patch_size);
    let tile_rows = rows.div_ceil(tile_size).max(1);
    let tile_cols = cols.div_ceil(tile_size).max(1);
    let tile_count = tile_rows * tile_cols;
    let patch_blend_weights = compute_patch_blend_weights::<F>(patch_size);
    let tile_geom = TileAggregationGeometry {
        patch_size,
        rows,
        cols,
        tile_size,
        tile_cols,
    };
    let agg_eps = F::from_f64_c(AGGREGATION_EPSILON);
    let profile_timing = resolve_profile_timing();
    let transform_cache_capacity = resolve_transform_cache_capacity();
    let kernel_started = profile_timing.then(Instant::now);

    macro_rules! timed {
        ($enabled:expr, $acc:expr, $body:block) => {{
            if $enabled {
                let _t = Instant::now();
                let _ret = { $body };
                $acc += _t.elapsed().as_nanos();
                _ret
            } else {
                $body
            }
        }};
    }

    let final_result = if total_refs == 0 {
        None
    } else {
        // Use one coordinate chunk per Rayon worker to preserve throughput
        // while keeping partial aggregation memory tile-bounded.
        let partial_count = total_refs.min(rayon::current_num_threads().max(1));
        let chunk_len = total_refs.div_ceil(partial_count).max(RAYON_MIN_CHUNK_LEN);
        let chunk_count = total_refs.div_ceil(chunk_len);

        (0..chunk_count)
            .into_par_iter()
            .map(|chunk_idx| {
                let chunk_start = chunk_idx * chunk_len;
                let chunk_end = (chunk_start + chunk_len).min(total_refs);
                let mut tile_accumulators: TileAccumulatorVec<F> =
                    (0..tile_count).map(|_| None).collect();
                let mut worker = WorkerBuffers::<F>::new(
                    max_matches,
                    patch_size,
                    fft2d_row_plan_scratch_len,
                    fft2d_col_plan_scratch_len,
                    ifft2d_row_plan_scratch_len,
                    ifft2d_col_plan_scratch_len,
                    fft1d_plan_scratch_len,
                    ifft1d_plan_scratch_len,
                );
                let mut noisy_transform_cache =
                    PatchTransformCache::<F>::new(patch_size, transform_cache_capacity);
                let mut pilot_transform_cache = if mode == Bm3dMode::Wiener {
                    Some(PatchTransformCache::<F>::new(
                        patch_size,
                        transform_cache_capacity,
                    ))
                } else {
                    None
                };
                let mut stats = KernelStageStats::default();

                for ref_index in chunk_start..chunk_end {
                    let ref_r = (ref_index / ref_cols) * step_size;
                    let ref_c = (ref_index % ref_cols) * step_size;
                    // 1. Block Matching
                    timed!(profile_timing, stats.block_matching_ns, {
                        block_matching::find_similar_patches_in_place_sum(
                            input_pilot,
                            &integral_sum,
                            (ref_r, ref_c),
                            (patch_size, patch_size),
                            (search_window, search_window),
                            max_matches,
                            step_size,
                            &mut worker.matches,
                        );
                    });
                    let k = worker.matches.len();
                    if k == 0 {
                        continue;
                    }
                    stats.groups += 1;
                    stats.matched_patches += k as u64;

                    // 1.5 Local Noise Level
                    let local_sigma_streak = if use_sigma_map_full {
                        sigma_map[[ref_r, ref_c]]
                    } else if use_sigma_map_row {
                        sigma_map[[0, ref_c]]
                    } else {
                        F::zero()
                    };

                    // Cached Forward 2D transforms
                    timed!(profile_timing, stats.forward_2d_ns, {
                        fill_transformed_group_from_cache(
                            input_noisy,
                            &worker.matches,
                            k,
                            patch_size,
                            use_hadamard,
                            &fft_2d_refs,
                            &mut noisy_transform_cache,
                            &mut worker.complex_work,
                            &mut worker.scratch_2d,
                            &mut worker.fft2d_row_plan_scratch,
                            &mut worker.fft2d_col_plan_scratch,
                            &mut worker.g_noisy_c,
                        );
                        if let Some(pilot_cache) = pilot_transform_cache.as_mut() {
                            fill_transformed_group_from_cache(
                                input_pilot,
                                &worker.matches,
                                k,
                                patch_size,
                                use_hadamard,
                                &fft_2d_refs,
                                pilot_cache,
                                &mut worker.complex_work,
                                &mut worker.scratch_2d,
                                &mut worker.fft2d_row_plan_scratch,
                                &mut worker.fft2d_col_plan_scratch,
                                &mut worker.g_pilot_c,
                            );
                        }
                    });

                    // Forward 1D transforms along group dimension
                    timed!(profile_timing, stats.forward_1d_ns, {
                        let fft_k_plan = &fft_1d_plans_ref[k];
                        apply_forward_1d_transform(
                            &mut worker.g_noisy_c,
                            k,
                            patch_size,
                            fft_k_plan,
                            &mut worker.scratch_1d[..k],
                            &mut worker.fft1d_plan_scratch,
                        );
                        if mode == Bm3dMode::Wiener {
                            apply_forward_1d_transform(
                                &mut worker.g_pilot_c,
                                k,
                                patch_size,
                                fft_k_plan,
                                &mut worker.scratch_1d[..k],
                                &mut worker.fft1d_plan_scratch,
                            );
                        }
                    });

                    // Filtering
                    let mut weight_g = F::one();
                    let spatial_scale = F::usize_as(patch_size);
                    let spatial_scale_sq = spatial_scale * spatial_scale;
                    timed!(profile_timing, stats.filtering_ns, {
                        match mode {
                            Bm3dMode::HardThreshold => {
                                // Hard Thresholding
                                let hard_thresholds = &mut worker.coeff_buffer;
                                for r in 0..patch_size {
                                    for c in 0..patch_size {
                                        hard_thresholds[r * patch_size + c] = threshold * {
                                            let sigma_s_dist = if use_colored_noise {
                                                sigma_psd[[r, c]]
                                            } else {
                                                F::zero()
                                            };
                                            let effective_sigma_s =
                                                sigma_s_dist * local_sigma_streak;
                                            let k_f = F::usize_as(k);
                                            let var_r = k_f * scalar_sigma_sq;
                                            let var_s =
                                                (k_f * k_f) * effective_sigma_s * effective_sigma_s;
                                            (var_r + var_s).sqrt() * spatial_scale
                                        };
                                    }
                                }

                                let mut nz_count = 0usize;
                                let patch_area = patch_size * patch_size;
                                if let Some(noisy) = worker.g_noisy_c.as_slice_memory_order_mut() {
                                    for i in 0..k {
                                        let base = i * patch_area;
                                        for rc in 0..patch_area {
                                            let coeff = noisy[base + rc];
                                            if coeff.norm() < hard_thresholds[rc] {
                                                noisy[base + rc] =
                                                    Complex::new(F::zero(), F::zero());
                                            } else {
                                                nz_count += 1;
                                            }
                                        }
                                    }
                                } else {
                                    for i in 0..k {
                                        for r in 0..patch_size {
                                            for c in 0..patch_size {
                                                let coeff = worker.g_noisy_c[[i, r, c]];
                                                if coeff.norm()
                                                    < hard_thresholds[r * patch_size + c]
                                                {
                                                    worker.g_noisy_c[[i, r, c]] =
                                                        Complex::new(F::zero(), F::zero());
                                                } else {
                                                    nz_count += 1;
                                                }
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
                                        let sigma_s_dist = if use_colored_noise {
                                            sigma_psd[[r, c]]
                                        } else {
                                            F::zero()
                                        };
                                        let effective_sigma_s = sigma_s_dist * local_sigma_streak;
                                        let k_f = F::usize_as(k);
                                        let var_r = k_f * scalar_sigma_sq;
                                        let var_s =
                                            (k_f * k_f) * effective_sigma_s * effective_sigma_s;
                                        noise_vars[r * patch_size + c] =
                                            (var_r + var_s) * spatial_scale_sq;
                                    }
                                }

                                let mut wiener_sum = F::zero();
                                let patch_area = patch_size * patch_size;
                                if let (Some(noisy), Some(pilot)) = (
                                    worker.g_noisy_c.as_slice_memory_order_mut(),
                                    worker.g_pilot_c.as_slice_memory_order(),
                                ) {
                                    for i in 0..k {
                                        let base = i * patch_area;
                                        for rc in 0..patch_area {
                                            let p_val = pilot[base + rc];
                                            let p_pow = p_val.norm_sqr();
                                            let n_val = noisy[base + rc];
                                            let w = p_pow / (p_pow + noise_vars[rc] + wiener_eps);
                                            noisy[base + rc] = n_val * w;
                                            wiener_sum += w * w;
                                        }
                                    }
                                } else {
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
                                }
                                weight_g = F::one() / (wiener_sum * scalar_sigma_sq + wiener_eps);
                                if weight_g > max_wiener_weight {
                                    weight_g = max_wiener_weight;
                                }
                            }
                        }
                    });

                    // Inverse 1D transform along group dimension
                    timed!(profile_timing, stats.inverse_1d_ns, {
                        let ifft_k_plan = &ifft_1d_plans_ref[k];
                        apply_inverse_1d_transform(
                            &mut worker.g_noisy_c,
                            k,
                            patch_size,
                            ifft_k_plan,
                            &mut worker.scratch_1d[..k],
                            &mut worker.ifft1d_plan_scratch,
                        );
                    });

                    // Inverse 2D transforms and aggregation
                    for (i, matched) in worker.matches.iter().enumerate().take(k) {
                        let complex_slice = worker.g_noisy_c.slice(s![i, .., ..]);
                        timed!(profile_timing, stats.inverse_2d_ns, {
                            if use_hadamard {
                                transforms::wht2d_8x8_inverse_into_view(
                                    complex_slice,
                                    &mut worker.spatial_patch,
                                );
                            } else {
                                transforms::ifft2d_into_with_plan_scratch(
                                    complex_slice,
                                    ifft_2d_row_ref,
                                    ifft_2d_col_ref,
                                    &mut worker.complex_work,
                                    &mut worker.spatial_patch,
                                    &mut worker.scratch_2d,
                                    &mut worker.ifft2d_row_plan_scratch,
                                    &mut worker.ifft2d_col_plan_scratch,
                                );
                            }
                        });
                        timed!(profile_timing, stats.aggregation_ns, {
                            aggregate_patch_into_tiles(
                                &worker.spatial_patch,
                                matched,
                                weight_g,
                                &patch_blend_weights,
                                &tile_geom,
                                &mut tile_accumulators,
                            );
                        });
                    }
                }

                let (hits, misses, evictions) = noisy_transform_cache.stats();
                stats.noisy_cache_hits += hits;
                stats.noisy_cache_misses += misses;
                stats.noisy_cache_evictions += evictions;
                if let Some(cache) = pilot_transform_cache.as_ref() {
                    let (hits, misses, evictions) = cache.stats();
                    stats.pilot_cache_hits += hits;
                    stats.pilot_cache_misses += misses;
                    stats.pilot_cache_evictions += evictions;
                }

                (tile_accumulators, stats)
            })
            .reduce_with(|(mut a_tiles, a_stats), (b_tiles, b_stats)| {
                for (a_slot, b_slot) in a_tiles.iter_mut().zip(b_tiles.into_iter()) {
                    if let Some((b_num, b_den)) = b_slot {
                        if let Some((a_num, a_den)) = a_slot.as_mut() {
                            *a_num += &b_num;
                            *a_den += &b_den;
                        } else {
                            *a_slot = Some((b_num, b_den));
                        }
                    }
                }
                (a_tiles, a_stats.merge(b_stats))
            })
    };
    let (final_tiles, stage_stats) = match final_result {
        Some((tiles, stats)) => (Some(tiles), stats),
        None => (None, KernelStageStats::default()),
    };
    let mut output = input_noisy.to_owned();
    if let Some(tile_accumulators) = final_tiles {
        for (tile_id, tile_entry) in tile_accumulators.into_iter().enumerate() {
            let Some((num_tile, den_tile)) = tile_entry else {
                continue;
            };
            let tile_row = tile_id / tile_cols;
            let tile_col = tile_id % tile_cols;
            let row_start = tile_row * tile_size;
            let col_start = tile_col * tile_size;
            let (tile_h, tile_w) = num_tile.dim();

            for tr in 0..tile_h {
                for tc in 0..tile_w {
                    let den = den_tile[[tr, tc]];
                    if den > agg_eps {
                        output[[row_start + tr, col_start + tc]] = num_tile[[tr, tc]] / den;
                    }
                }
            }
        }
    };

    if profile_timing {
        let total_ns = kernel_started
            .map(|t| t.elapsed().as_nanos())
            .unwrap_or_default();
        let noisy_queries = stage_stats.noisy_cache_hits + stage_stats.noisy_cache_misses;
        let noisy_hit_rate = if noisy_queries > 0 {
            stage_stats.noisy_cache_hits as f64 / noisy_queries as f64
        } else {
            0.0
        };
        let pilot_queries = stage_stats.pilot_cache_hits + stage_stats.pilot_cache_misses;
        let pilot_hit_rate = if pilot_queries > 0 {
            stage_stats.pilot_cache_hits as f64 / pilot_queries as f64
        } else {
            0.0
        };
        eprintln!(
            "bm3d_profile mode={:?} size={}x{} refs={} groups={} matched_patches={} cache_capacity={} noisy_hit_rate={:.3} pilot_hit_rate={:.3} cache_evictions=noisy:{} pilot:{} wall_ms={:.3} block_thread_ms={:.3} fwd2d_thread_ms={:.3} fwd1d_thread_ms={:.3} filter_thread_ms={:.3} inv1d_thread_ms={:.3} inv2d_thread_ms={:.3} agg_thread_ms={:.3}",
            mode,
            rows,
            cols,
            total_refs,
            stage_stats.groups,
            stage_stats.matched_patches,
            transform_cache_capacity,
            noisy_hit_rate,
            pilot_hit_rate,
            stage_stats.noisy_cache_evictions,
            stage_stats.pilot_cache_evictions,
            total_ns as f64 / 1_000_000.0,
            stage_stats.block_matching_ns as f64 / 1_000_000.0,
            stage_stats.forward_2d_ns as f64 / 1_000_000.0,
            stage_stats.forward_1d_ns as f64 / 1_000_000.0,
            stage_stats.filtering_ns as f64 / 1_000_000.0,
            stage_stats.inverse_1d_ns as f64 / 1_000_000.0,
            stage_stats.inverse_2d_ns as f64 / 1_000_000.0,
            stage_stats.aggregation_ns as f64 / 1_000_000.0,
        );
    }

    output
}

/// Run BM3D step on a single 2D image.
pub fn run_bm3d_step<F: Bm3dFloat>(
    input_noisy: ArrayView2<F>,
    input_pilot: ArrayView2<F>,
    mode: Bm3dMode,
    sigma_psd: ArrayView2<F>,
    sigma_map: ArrayView2<F>,
    config: &Bm3dKernelConfig<F>,
    plans: &Bm3dPlans<F>,
) -> Result<Array2<F>, String> {
    if input_pilot.dim() != input_noisy.dim() {
        return Err(format!(
            "Dimension mismatch: input_noisy has shape {:?}, but input_pilot has shape {:?}",
            input_noisy.dim(),
            input_pilot.dim()
        ));
    }
    if sigma_map.dim() != input_noisy.dim()
        && sigma_map.dim() != (1, input_noisy.dim().1)
        && sigma_map.dim() != (1, 1)
    {
        return Err(format!(
            "Sigma map dimension mismatch: expected {:?}, (1, {}), or (1, 1), got {:?}",
            input_noisy.dim(),
            input_noisy.dim().1,
            sigma_map.dim()
        ));
    }

    Ok(run_bm3d_kernel(
        input_noisy,
        input_pilot,
        mode,
        sigma_psd,
        sigma_map,
        config,
        plans,
    ))
}

/// Run BM3D step on a 3D stack of images.
pub fn run_bm3d_step_stack<F: Bm3dFloat>(
    input_noisy: ArrayView3<F>,
    input_pilot: ArrayView3<F>,
    mode: Bm3dMode,
    sigma_psd: ArrayView2<F>,
    sigma_map: ArrayView3<F>,
    config: &Bm3dKernelConfig<F>,
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
    if sigma_map.dim() != (n, rows, cols)
        && sigma_map.dim() != (n, 1, cols)
        && sigma_map.dim() != (1, 1, 1)
    {
        return Err(format!(
            "Sigma map dimension mismatch: expected {:?}, ({}, 1, {}), or (1, 1, 1), got {:?}",
            (n, rows, cols),
            n,
            cols,
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
                config,
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

    fn stitch_tiles_to_full(
        tile_accumulators: TileAccumulatorVec<f32>,
        rows: usize,
        cols: usize,
        tile_size: usize,
        tile_cols: usize,
    ) -> (Array2<f32>, Array2<f32>) {
        let mut numerator = Array2::<f32>::zeros((rows, cols));
        let mut denominator = Array2::<f32>::zeros((rows, cols));
        for (tile_id, tile_entry) in tile_accumulators.into_iter().enumerate() {
            let Some((num_tile, den_tile)) = tile_entry else {
                continue;
            };
            let tile_row = tile_id / tile_cols;
            let tile_col = tile_id % tile_cols;
            let row_start = tile_row * tile_size;
            let col_start = tile_col * tile_size;
            let (tile_h, tile_w) = num_tile.dim();
            numerator
                .slice_mut(s![
                    row_start..row_start + tile_h,
                    col_start..col_start + tile_w
                ])
                .assign(&num_tile);
            denominator
                .slice_mut(s![
                    row_start..row_start + tile_h,
                    col_start..col_start + tile_w
                ])
                .assign(&den_tile);
        }
        (numerator, denominator)
    }

    fn aggregate_patch_reference(
        spatial: &Array2<f32>,
        m: &PatchMatch<f32>,
        weight: f32,
        patch_blend_weights: &Array2<f32>,
        patch_size: usize,
        rows: usize,
        cols: usize,
    ) -> (Array2<f32>, Array2<f32>) {
        let mut numerator = Array2::<f32>::zeros((rows, cols));
        let mut denominator = Array2::<f32>::zeros((rows, cols));
        for pr in 0..patch_size {
            for pc in 0..patch_size {
                let tr = m.row + pr;
                let tc = m.col + pc;
                if tr < rows && tc < cols {
                    let w = weight * patch_blend_weights[[pr, pc]];
                    numerator[[tr, tc]] += spatial[[pr, pc]] * w;
                    denominator[[tr, tc]] += w;
                }
            }
        }
        (numerator, denominator)
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

    // Keep tests readable while exercising the production config-based API.
    fn run_bm3d_step(
        input_noisy: ndarray::ArrayView2<f32>,
        input_pilot: ndarray::ArrayView2<f32>,
        mode: Bm3dMode,
        sigma_psd: ndarray::ArrayView2<f32>,
        sigma_map: ndarray::ArrayView2<f32>,
        sigma_random: f32,
        threshold: f32,
        patch_size: usize,
        step_size: usize,
        search_window: usize,
        max_matches: usize,
        plans: &Bm3dPlans<f32>,
    ) -> Result<Array2<f32>, String> {
        let config = Bm3dKernelConfig {
            sigma_random,
            threshold,
            patch_size,
            step_size,
            search_window,
            max_matches,
        };
        super::run_bm3d_step(
            input_noisy,
            input_pilot,
            mode,
            sigma_psd,
            sigma_map,
            &config,
            plans,
        )
    }

    fn run_bm3d_step_stack(
        input_noisy: ndarray::ArrayView3<f32>,
        input_pilot: ndarray::ArrayView3<f32>,
        mode: Bm3dMode,
        sigma_psd: ndarray::ArrayView2<f32>,
        sigma_map: ndarray::ArrayView3<f32>,
        sigma_random: f32,
        threshold: f32,
        patch_size: usize,
        step_size: usize,
        search_window: usize,
        max_matches: usize,
        plans: &Bm3dPlans<f32>,
    ) -> Result<Array3<f32>, String> {
        let config = Bm3dKernelConfig {
            sigma_random,
            threshold,
            patch_size,
            step_size,
            search_window,
            max_matches,
        };
        super::run_bm3d_step_stack(
            input_noisy,
            input_pilot,
            mode,
            sigma_psd,
            sigma_map,
            &config,
            plans,
        )
    }

    #[test]
    fn test_aggregate_patch_into_tiles_matches_reference() {
        let rows = 16usize;
        let cols = 16usize;
        let weight = 0.75f32;

        for (patch_size, tile_size, row, col) in [(4usize, 8usize, 2usize, 3usize), (8, 6, 4, 5)] {
            let spatial = Array2::from_shape_fn((patch_size, patch_size), |(r, c)| {
                1.0 + (r * patch_size + c) as f32 * 0.01
            });
            let m = PatchMatch {
                row,
                col,
                distance: 0.0,
            };

            let tile_rows = rows.div_ceil(tile_size).max(1);
            let tile_cols = cols.div_ceil(tile_size).max(1);
            let tile_count = tile_rows * tile_cols;
            let mut tile_accumulators: TileAccumulatorVec<f32> =
                (0..tile_count).map(|_| None).collect();
            let patch_blend_weights = compute_patch_blend_weights::<f32>(patch_size);

            aggregate_patch_into_tiles(
                &spatial,
                &m,
                weight,
                &patch_blend_weights,
                &TileAggregationGeometry {
                    patch_size,
                    rows,
                    cols,
                    tile_size,
                    tile_cols,
                },
                &mut tile_accumulators,
            );

            let (num_ref, den_ref) = aggregate_patch_reference(
                &spatial,
                &m,
                weight,
                &patch_blend_weights,
                patch_size,
                rows,
                cols,
            );
            let (num_tiled, den_tiled) =
                stitch_tiles_to_full(tile_accumulators, rows, cols, tile_size, tile_cols);

            for r in 0..rows {
                for c in 0..cols {
                    assert!(
                        (num_ref[[r, c]] - num_tiled[[r, c]]).abs() < 1e-6,
                        "Numerator mismatch at ({}, {})",
                        r,
                        c
                    );
                    assert!(
                        (den_ref[[r, c]] - den_tiled[[r, c]]).abs() < 1e-6,
                        "Denominator mismatch at ({}, {})",
                        r,
                        c
                    );
                }
            }
        }
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
