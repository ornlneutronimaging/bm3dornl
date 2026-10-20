//! Multi-Scale BM3D Streak Removal
//!
//! This module implements the multi-scale BM3D algorithm from:
//! Mäkinen, Y., Marchesini, S., Foi, A., 2021,
//! "Ring artifact reduction via multiscale nonlocal collaborative filtering
//! of spatially correlated noise"
//! J. Synchrotron Rad. 28(3). DOI: http://doi.org/10.1107/S1600577521001910
//!
//! The algorithm works by:
//! 1. Building a horizontal pyramid via sum-convolution binning
//! 2. Processing coarse-to-fine, propagating denoised residuals
//! 3. Using cubic spline interpolation for debinning

use crate::utils::compute_1d_median_filter;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

use crate::float_trait::Bm3dFloat;
use crate::orchestration::{Bm3dConfig, RingRemovalMode, bm3d_ring_artifact_removal};
use crate::pipeline::check_image_size;

const MIN_SCALE_WIDTH: usize = 40;

const BINNING_FACTOR: usize = 2;

const DEFAULT_DEBIN_ITERATIONS: usize = 30;

const DEFAULT_MULTISCALE_FILTER_STRENGTH: f64 = 1.0;

const DEFAULT_MULTISCALE_THRESHOLD: f64 = 3.5;

const VERTICAL_BIN_TARGET_ROWS: usize = 240;

#[derive(Debug, Clone)]
pub struct MultiscaleConfig<F: Bm3dFloat> {
    /// Override automatic scale calculation
    pub num_scales: Option<usize>,
    /// Multiplier for BM3D filtering intensity. Default: 1.0
    pub filter_strength: F,
    /// Hard threshold coefficient. Default: 3.5
    pub threshold: F,
    /// Iterations for cubic spline debinning. Default: 30
    pub debin_iterations: usize,
    pub log_domain_input: bool,
    /// Inner BM3D config (patch_size, search_window, etc.)
    pub bm3d_config: Bm3dConfig<F>,
}

impl<F: Bm3dFloat> Default for MultiscaleConfig<F> {
    fn default() -> Self {
        let bm3d_config = Bm3dConfig {
            threshold: F::from_f64_c(DEFAULT_MULTISCALE_THRESHOLD),
            filter_strength: F::from_f64_c(DEFAULT_MULTISCALE_FILTER_STRENGTH),
            ..Default::default()
        };

        Self {
            num_scales: None,
            filter_strength: F::from_f64_c(DEFAULT_MULTISCALE_FILTER_STRENGTH),
            threshold: F::from_f64_c(DEFAULT_MULTISCALE_THRESHOLD),
            debin_iterations: DEFAULT_DEBIN_ITERATIONS,
            log_domain_input: false,
            bm3d_config,
        }
    }
}

impl<F: Bm3dFloat> MultiscaleConfig<F> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.filter_strength <= F::zero() {
            return Err("filter_strength must be > 0".to_string());
        }
        if self.threshold < F::zero() {
            return Err("threshold must be >= 0".to_string());
        }
        if self.debin_iterations == 0 {
            return Err("debin_iterations must be > 0".to_string());
        }
        self.bm3d_config.validate()
    }
}

pub fn compute_num_scales(width: usize) -> usize {
    if width <= MIN_SCALE_WIDTH {
        return 0;
    }
    let ratio = width as f64 / MIN_SCALE_WIDTH as f64;
    let k = ratio.log2().floor().max(0.0) as usize;
    k.min(3)
}

fn symmetric_pad_right<F: Bm3dFloat>(arr: ArrayView2<F>, pad_amount: usize) -> Array2<F> {
    if pad_amount == 0 {
        return arr.to_owned();
    }

    let (rows, cols) = arr.dim();
    let new_cols = cols + pad_amount;
    let mut result = Array2::zeros((rows, new_cols));

    result.slice_mut(s![.., ..cols]).assign(&arr);

    for p in 0..pad_amount {
        let src_col = cols - 1 - (p % cols);
        for r in 0..rows {
            result[[r, cols + p]] = arr[[r, src_col]];
        }
    }

    result
}

/// Bin a 2D array horizontally using sum-convolution.
pub fn bin_horizontal<F: Bm3dFloat>(arr: ArrayView2<F>, factor: usize) -> Array2<F> {
    if factor <= 1 {
        return arr.to_owned();
    }

    let (rows, cols) = arr.dim();

    let mod_pad = factor - ((cols - 1) % factor) - 1;
    let padded = symmetric_pad_right(arr, mod_pad);
    let padded_cols = padded.ncols();

    let mut convolved = Array2::zeros((rows, padded_cols));
    for r in 0..rows {
        for c in 0..padded_cols {
            let mut sum = F::zero();
            for k in 0..factor {
                let src_c = c as isize + k as isize - (factor as isize / 2);
                if src_c >= 0 && (src_c as usize) < padded_cols {
                    sum += padded[[r, src_c as usize]];
                }
            }
            convolved[[r, c]] = sum;
        }
    }

    let h_half = factor / 2;
    let start = h_half + if factor % 2 == 1 { 1 } else { 0 };
    let end = padded_cols - h_half + 1;

    let num_bins = (end - start).div_ceil(factor);
    let mut result = Array2::zeros((rows, num_bins));

    for r in 0..rows {
        for (i, c) in (start..end).step_by(factor).enumerate() {
            if i < num_bins && c < padded_cols {
                result[[r, i]] = convolved[[r, c]];
            }
        }
    }

    result
}

fn bin_vertical_mean<F: Bm3dFloat>(arr: ArrayView2<F>, factor: usize) -> Array2<F> {
    if factor <= 1 {
        return arr.to_owned();
    }
    let (rows, cols) = arr.dim();
    let binned_rows = rows.div_ceil(factor);
    let mut result = Array2::zeros((binned_rows, cols));
    for br in 0..binned_rows {
        let r_start = br * factor;
        let r_end = (r_start + factor).min(rows);
        let inv_count = F::one() / F::from_f64_c((r_end - r_start) as f64);
        for c in 0..cols {
            let mut sum = F::zero();
            for r in r_start..r_end {
                sum += arr[[r, c]];
            }
            result[[br, c]] = sum * inv_count;
        }
    }
    result
}

fn compute_bin_weights<F: Bm3dFloat>(target_width: usize, factor: usize) -> Array1<F> {
    if factor <= 1 {
        return Array1::ones(target_width);
    }

    let ones = Array2::from_elem((1, target_width), F::one());
    let mod_pad = factor - ((target_width - 1) % factor) - 1;
    let padded = symmetric_pad_right(ones.view(), mod_pad);
    let padded_cols = padded.ncols();

    let mut convolved = Array1::zeros(padded_cols);
    for c in 0..padded_cols {
        let mut sum = F::zero();
        for k in 0..factor {
            let src_c = c as isize + k as isize - (factor as isize / 2);
            if src_c >= 0 && (src_c as usize) < padded_cols {
                sum += padded[[0, src_c as usize]];
            }
        }
        convolved[c] = sum;
    }

    convolved.slice(s![..target_width]).to_owned()
}

fn cubic_spline_interpolate_row<F: Bm3dFloat>(
    x_coords: &[f64],
    y_values: ArrayView1<F>,
    target_x: &[f64],
    output: &mut [F],
) {
    let len = x_coords.len();
    if len < 2 {
        output.fill(F::zero());
        return;
    }

    // t is between 0 and 1 (distance between p1 and p2)
    let catmull_rom = |p0: f64, p1: f64, p2: f64, p3: f64, t: f64| -> f64 {
        let t2 = t * t;
        let t3 = t2 * t;
        0.5 * ((2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
    };

    let mut last_idx = 0;

    for (out_val, &tx) in output.iter_mut().zip(target_x.iter()) {
        let tx = tx.clamp(x_coords[0], x_coords[len - 1]);

        // Find index i such that x_coords[i] <= tx < x_coords[i+1]
        while last_idx < len - 2 && x_coords[last_idx + 1] <= tx {
            last_idx += 1;
        }
        let i = last_idx;

        let p1_idx = i;
        let p2_idx = (i + 1).min(len - 1);
        let p0_idx = i.saturating_sub(1);
        let p3_idx = (i + 2).min(len - 1);

        let x1 = x_coords[p1_idx];
        let x2 = x_coords[p2_idx];
        let dist = x2 - x1;

        let val = if dist.abs() < 1e-6 {
            y_values[p1_idx].to_f64().unwrap_or(0.0)
        } else {
            let t = (tx - x1) / dist;

            let y0 = y_values[p0_idx].to_f64().unwrap_or(0.0);
            let y1 = y_values[p1_idx].to_f64().unwrap_or(0.0);
            let y2 = y_values[p2_idx].to_f64().unwrap_or(0.0);
            let y3 = y_values[p3_idx].to_f64().unwrap_or(0.0);

            catmull_rom(y0, y1, y2, y3, t)
        };

        *out_val = F::from_f64_c(val);
    }
}

/// Debin a 2D array horizontally using iterative cubic spline interpolation.
pub fn debin_horizontal<F: Bm3dFloat>(
    arr: ArrayView2<F>,
    target_width: usize,
    factor: usize,
    iterations: usize,
) -> Array2<F> {
    if factor <= 1 {
        return arr.to_owned();
    }

    let (rows, binned_cols) = arr.dim();
    let h_half = factor / 2;

    let n_counter = compute_bin_weights::<F>(target_width, factor);

    // Coordinates of bin centers in target space
    let x1c: Vec<usize> = (0..binned_cols)
        .map(|i| h_half + (if factor % 2 == 1 { 1 } else { 0 }) + i * factor)
        .filter(|&c| c < target_width)
        .collect();

    let x1: Vec<f64> = (0..binned_cols)
        .map(|i| {
            (h_half as f64) + 1.0 - (if factor.is_multiple_of(2) { 0.5 } else { 0.0 })
                + (i * factor) as f64
        })
        .collect();

    let ix1: Vec<f64> = (1..=target_width).map(|i| i as f64).collect();

    let mut y_j = Array2::zeros((rows, target_width));

    let mut row_buffer = vec![F::zero(); target_width];

    for iter in 0..iterations.max(1) {
        let r_j = if iter > 0 {
            let binned_y_j = bin_horizontal(y_j.view(), factor);
            let min_cols = arr.ncols().min(binned_y_j.ncols());
            let mut residual = Array2::zeros((rows, min_cols));
            for r in 0..rows {
                for c in 0..min_cols {
                    residual[[r, c]] = arr[[r, c]] - binned_y_j[[r, c]];
                }
            }
            residual
        } else {
            arr.to_owned()
        };

        let mut normalized = Array2::zeros((rows, r_j.ncols()));
        for r in 0..rows {
            for (c, &idx) in x1c.iter().enumerate() {
                if c < r_j.ncols() && idx < target_width {
                    let weight = n_counter[idx];
                    if weight > F::zero() {
                        normalized[[r, c]] = r_j[[r, c]] / weight;
                    }
                }
            }
        }

        let x1_trimmed: Vec<f64> = x1.iter().take(normalized.ncols()).copied().collect();
        if x1_trimmed.len() >= 2 {
            for r in 0..rows {
                let row_values = normalized.row(r);
                let row_slice = row_values.slice(s![..x1_trimmed.len()]);

                cubic_spline_interpolate_row(&x1_trimmed, row_slice, &ix1, &mut row_buffer);

                for (c, &val) in row_buffer.iter().enumerate() {
                    if c < target_width {
                        y_j[[r, c]] += val;
                    }
                }
            }
        }
    }

    y_j
}

pub fn generate_psd_shapes<F: Bm3dFloat>(denoise_sizes: &[usize]) -> Vec<Array1<F>> {
    let mut psd_shapes = Vec::with_capacity(denoise_sizes.len());

    let kernel_half_vec = compute_residual_kernel::<f64>();

    for i in 0..denoise_sizes.len() {
        let sz = denoise_sizes[i];

        if i < denoise_sizes.len() - 1 {
            let kernel_half_len = kernel_half_vec.len();
            let full_kernel_len = kernel_half_len * 2 - 1;

            let trim_amount = if full_kernel_len > sz {
                (full_kernel_len - sz) / 2
            } else {
                0
            };
            let trimmed_start = trim_amount.min(kernel_half_len - 1);
            let trimmed_half: Vec<f64> = kernel_half_vec[trimmed_start..].to_vec();

            let mut residual_kernel: Vec<f64> = trimmed_half.clone();
            for j in (0..trimmed_half.len() - 1).rev() {
                residual_kernel.push(trimmed_half[j]);
            }

            let psd = compute_fft_psd::<F>(&residual_kernel, sz);
            psd_shapes.push(psd);
        } else {
            // Coarsest scale: flat PSD (white noise assumption)
            let scale = F::from_f64_c(1.0 / (sz as f64).sqrt());
            psd_shapes.push(Array1::from_elem(sz, scale));
        }
    }

    psd_shapes
}

/// Compute |FFT(kernel)|² padded/truncated to target size.
fn compute_fft_psd<F: Bm3dFloat>(kernel: &[f64], target_size: usize) -> Array1<F> {
    use rustfft::{FftPlanner, num_complex::Complex};

    let mut padded = vec![Complex::new(0.0, 0.0); target_size];
    for (i, &val) in kernel.iter().enumerate() {
        if i < target_size {
            padded[i] = Complex::new(val, 0.0);
        }
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(target_size);
    fft.process(&mut padded);

    Array1::from_iter(padded.iter().map(|c| F::from_f64_c(c.norm_sqr())))
}

/// Overlapping full-height segments used to model the horizontal
/// nonstationarity of the streak noise.
const SIGMA_SEGMENTS: usize = 16;

/// Narrowest segment worth estimating on. Below this the MAD has too few
/// samples per row to be stable.
const MIN_SEGMENT_COLS: usize = 24;

/// Segments quieter than this fraction of the loudest segment are air, and
/// are excluded when reducing the profile to a single sigma.
const ACTIVE_SEGMENT_RATIO: f64 = 0.1;

fn estimate_sigma_profile<F: Bm3dFloat>(sinogram: ArrayView2<F>) -> Vec<F> {
    let (rows, cols) = sinogram.dim();
    if rows < 16 || cols == 0 {
        return vec![F::zero(); cols];
    }

    let mut lo_v = f64::INFINITY;
    let mut hi_v = f64::NEG_INFINITY;
    for v in sinogram.iter() {
        let x = v.to_f64().unwrap_or(0.0);
        if x < lo_v {
            lo_v = x;
        }
        if x > hi_v {
            hi_v = x;
        }
    }
    if hi_v <= lo_v {
        return vec![F::zero(); cols];
    }

    let filtered = {
        let smoothed = crate::noise_estimation::gaussian_filter_1d_vertical_recursive(
            sinogram,
            rows as f64 / 12.0,
        );
        let db3: Vec<F> = DB3_ANALYSIS_HI
            .iter()
            .map(|&x| F::from_f64_c(x))
            .collect::<Vec<_>>();
        convolve_1d_horizontal_internal(smoothed.view(), &db3)
    };

    let margin = (DB3_ANALYSIS_HI.len() - 1).min(cols / 2);
    let (lo, hi) = (margin, cols - margin);
    if hi <= lo {
        return vec![F::zero(); cols];
    }

    let usable = hi - lo;
    let seg_width = (usable / SIGMA_SEGMENTS).max(MIN_SEGMENT_COLS).min(usable);
    let step = (seg_width / 2).max(1);

    let mut profile = vec![F::zero(); cols];
    let mut start = lo;
    loop {
        let end = (start + seg_width).min(hi);
        let sigma =
            compute_mad_internal(filtered.slice(s![.., start..end])) * F::from_f64_c(MAD_TO_SIGMA);
        let assign_from = if start == lo { 0 } else { start };
        let assign_to = if end == hi { cols } else { end };
        for p in profile.iter_mut().take(assign_to).skip(assign_from) {
            if sigma > *p {
                *p = sigma;
            }
        }
        if end == hi {
            break;
        }
        start += step;
    }
    profile
}

fn estimate_noise_sigma_robust<F: Bm3dFloat>(sinogram: ArrayView2<F>) -> F {
    let profile = estimate_sigma_profile(sinogram);
    if profile.is_empty() {
        return F::zero();
    }
    let loudest = profile
        .iter()
        .copied()
        .fold(F::zero(), |a, b| if b > a { b } else { a });
    if loudest <= F::zero() {
        return F::zero();
    }
    let floor = loudest * F::from_f64_c(ACTIVE_SEGMENT_RATIO);
    let mut active: Vec<F> = profile.into_iter().filter(|&s| s >= floor).collect();
    if active.is_empty() {
        return loudest;
    }
    let mid = active.len() / 2;
    active.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    active[mid]
}

const DENOISE_SEGMENTS: usize = 8;

const DENOISE_SEGMENT_OVERLAP: f64 = 0.75;

const MIN_DENOISE_SEGMENT_WIDTH_FACTOR: usize = 4;

fn median_of<F: Bm3dFloat>(vals: &mut [F]) -> Option<F> {
    if vals.is_empty() {
        return None;
    }
    let mid = vals.len() / 2;
    vals.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    Some(vals[mid])
}

fn bm3d_denoise_segmented<F: Bm3dFloat>(
    image: ArrayView2<F>,
    config: &Bm3dConfig<F>,
    plans: &crate::pipeline::Bm3dPlans<F>,
    psd_x: Option<ArrayView1<F>>,
    requested_segments: usize,
    min_segment_width: usize,
) -> Result<Array2<F>, String> {
    let (rows, cols) = image.dim();
    let min_w = min_segment_width.max(config.patch_size).max(1);

    let v = DENOISE_SEGMENT_OVERLAP;
    let n_fit = if cols >= min_w {
        (1.0 + (cols as f64 / min_w as f64 - 1.0) / (1.0 - v)).floor() as usize
    } else {
        1
    };
    let n = requested_segments.clamp(1, n_fit.max(1));
    if n <= 1 {
        return crate::orchestration::bm3d_ring_artifact_removal_with_psd(
            image,
            RingRemovalMode::Streak,
            config,
            plans,
            psd_x,
        );
    }

    let seg_w = ((cols as f64 / (1.0 + (n as f64 - 1.0) * (1.0 - v))).ceil() as usize).min(cols);
    let last_start = cols - seg_w;

    // Per-column noise amplitude, measured once on this level; the air
    // exclusion floor is relative to the level's loudest column.
    let profile = estimate_sigma_profile(image);
    let loudest = profile
        .iter()
        .copied()
        .fold(F::zero(), |a, b| if b > a { b } else { a });
    let active_floor = loudest * F::from_f64_c(ACTIVE_SEGMENT_RATIO);
    let dust = F::from_f64_c(1.0e-9);

    let mut acc = Array2::<F>::zeros((rows, cols));
    let mut weight_sum = vec![F::zero(); cols];

    for i in 0..n {
        let start = (i * last_start + (n - 1) / 2) / (n - 1);
        let end = (start + seg_w).min(cols);
        let seg = image.slice(s![.., start..end]);

        let mut seg_config = config.clone();
        let mut vals: Vec<F> = profile[start..end]
            .iter()
            .copied()
            .filter(|&s| s >= active_floor)
            .collect();
        let seg_sigma = median_of(&mut vals).unwrap_or(F::zero());
        if seg_sigma > dust {
            seg_config.sigma_random = seg_sigma;
        }

        let den = crate::orchestration::bm3d_ring_artifact_removal_with_psd(
            seg,
            RingRemovalMode::Streak,
            &seg_config,
            plans,
            psd_x,
        )?;

        let width = end - start;
        for j in 0..width {
            let phase = std::f64::consts::PI * (j as f64 + 0.5) / width as f64;
            let w = F::from_f64_c(phase.sin() * phase.sin());
            let c = start + j;
            weight_sum[c] += w;
            for r in 0..rows {
                acc[[r, c]] += den[[r, j]] * w;
            }
        }
    }

    for c in 0..cols {
        let total = weight_sum[c];
        if total > F::zero() {
            for r in 0..rows {
                acc[[r, c]] /= total;
            }
        } else {
            for r in 0..rows {
                acc[[r, c]] = image[[r, c]];
            }
        }
    }
    Ok(acc)
}

const MAD_TO_SIGMA: f64 = 1.4826;

const DB3_ANALYSIS_HI: [f64; 6] = [
    0.03522629,
    -0.08544127,
    0.13501102,
    0.45987750,
    -0.80689151,
    0.33267055,
];

fn convolve_1d_horizontal_internal<F: Bm3dFloat>(data: ArrayView2<F>, kernel: &[F]) -> Array2<F> {
    let (rows, cols) = data.dim();
    let k_len = kernel.len();
    let radius = k_len / 2;
    let mut output = Array2::zeros((rows, cols));

    for r in 0..rows {
        for c in 0..cols {
            let mut sum = F::zero();
            for (k, &k_val) in kernel.iter().enumerate() {
                let k_idx = k as isize - radius as isize;
                let src_c = (c as isize + k_idx).clamp(0, (cols - 1) as isize);
                sum += data[[r, src_c as usize]] * k_val;
            }
            output[[r, c]] = sum;
        }
    }
    output
}

fn compute_mad_internal<F: Bm3dFloat>(data: ArrayView2<F>) -> F {
    let mut flat_data: Vec<F> = data.iter().cloned().collect();
    if flat_data.is_empty() {
        return F::zero();
    }

    let len = flat_data.len();
    let mid = len / 2;
    let (_, &mut median, _) =
        flat_data.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    let median_val = if len % 2 == 1 {
        median
    } else {
        let left = &flat_data[..mid];
        let prev = left
            .iter()
            .fold(F::neg_infinity(), |a, &b| if b > a { b } else { a });
        (prev + median) / F::from_f64_c(2.0)
    };

    let mut deviations: Vec<F> = flat_data.iter().map(|&x| (x - median_val).abs()).collect();
    let (_, &mut dev_median, _) =
        deviations.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    if len % 2 == 1 {
        dev_median
    } else {
        let left = &deviations[..mid];
        let prev = left
            .iter()
            .fold(F::neg_infinity(), |a, &b| if b > a { b } else { a });
        (prev + dev_median) / F::from_f64_c(2.0)
    }
}

pub fn multiscale_bm3d_streak_removal<F: Bm3dFloat>(
    sinogram: ArrayView2<F>,
    config: &MultiscaleConfig<F>,
) -> Result<Array2<F>, String> {
    let plans = crate::pipeline::Bm3dPlans::new(
        config.bm3d_config.patch_size,
        config.bm3d_config.max_matches,
    );
    multiscale_bm3d_streak_removal_with_plans(sinogram, config, &plans)
}

pub fn multiscale_bm3d_streak_removal_with_plans<F: Bm3dFloat>(
    sinogram: ArrayView2<F>,
    config: &MultiscaleConfig<F>,
    plans: &crate::pipeline::Bm3dPlans<F>,
) -> Result<Array2<F>, String> {
    config.validate()?;
    let (rows, cols) = sinogram.dim();
    check_image_size(rows, cols, config.bm3d_config.patch_size)?;

    if config.log_domain_input {
        return multiscale_denoise_log_domain(sinogram, config, plans);
    }

    let mut floor = F::infinity();
    let mut lo = F::infinity();
    let mut hi = F::neg_infinity();
    for &v in sinogram.iter() {
        if v > F::zero() && v < floor {
            floor = v;
        }
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    if !floor.is_finite() {
        if hi <= lo {
            return Ok(sinogram.to_owned());
        }
        return Err(
            "multiscale streak removal expects linear transmission data with at least one \
             positive value; if the input is already log-transformed, set \
             MultiscaleConfig::log_domain_input = true"
                .to_string(),
        );
    }

    let log_floor = floor.ln();
    let logged = sinogram.mapv(|v| if v > F::zero() { v.ln() } else { log_floor });
    let denoised = multiscale_denoise_log_domain(logged.view(), config, plans)?;
    Ok(denoised.mapv(|v| v.exp()))
}

fn multiscale_denoise_log_domain<F: Bm3dFloat>(
    sinogram: ArrayView2<F>,
    config: &MultiscaleConfig<F>,
    plans: &crate::pipeline::Bm3dPlans<F>,
) -> Result<Array2<F>, String> {
    let (rows, cols) = sinogram.dim();
    let factor = (rows / VERTICAL_BIN_TARGET_ROWS).max(1);
    if factor <= 1 {
        return multiscale_denoise_pyramid(sinogram, config, plans);
    }

    // Streaks are vertically constant and pass
    // through unchanged; per-pixel noise attenuates by sqrt(factor).
    let binned = bin_vertical_mean(sinogram, factor);
    let denoised_binned = multiscale_denoise_pyramid(binned.view(), config, plans)?;

    let correction = &denoised_binned - &binned;
    let mut profile = Vec::with_capacity(cols);
    for c in 0..cols {
        let mut col: Vec<F> = correction.column(c).to_vec();
        let mid = col.len() / 2;
        col.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        profile.push(col[mid]);
    }
    let mut output = sinogram.to_owned();
    for r in 0..rows {
        for c in 0..cols {
            output[[r, c]] += profile[c];
        }
    }
    Ok(output)
}

fn multiscale_denoise_pyramid<F: Bm3dFloat>(
    sinogram: ArrayView2<F>,
    config: &MultiscaleConfig<F>,
    plans: &crate::pipeline::Bm3dPlans<F>,
) -> Result<Array2<F>, String> {
    let (rows, cols) = sinogram.dim();

    let estimated_global = estimate_noise_sigma_robust(sinogram);

    let sigma_global = if config.bm3d_config.sigma_random > F::from_f64_c(1e-6) {
        config.bm3d_config.sigma_random
    } else {
        estimated_global
    };
    let noise_var_threshold = (sigma_global * sigma_global) * F::from_f64_c(2.0);

    let mut fine_col_vars = Vec::with_capacity(cols);
    let n_rows_f = F::from_f64_c(rows as f64);

    for c in 0..cols {
        let col = sinogram.column(c);

        let mut col_data: Vec<F> = col.to_vec();
        let mid = col_data.len() / 2;
        col_data.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let med = col_data[mid];

        let mut sum_sq = F::zero();
        for &val in col {
            let diff = val - med;
            sum_sq += diff * diff;
        }
        let var = sum_sq / n_rows_f;
        fine_col_vars.push(var);
    }

    let num_scales = config
        .num_scales
        .unwrap_or_else(|| compute_num_scales(cols));

    if num_scales == 0 {
        return bm3d_ring_artifact_removal(sinogram, RingRemovalMode::Streak, &config.bm3d_config);
    }

    let mut pyramid_orig: Vec<Array2<F>> = Vec::with_capacity(num_scales + 1);
    pyramid_orig.push(sinogram.to_owned());

    for _ in 0..num_scales {
        let last = pyramid_orig.last().unwrap();
        let binned = bin_horizontal(last.view(), BINNING_FACTOR);
        pyramid_orig.push(binned);
    }

    let mut pyramid_work: Vec<Array2<F>> = vec![Array2::zeros((0, 0)); num_scales + 1];
    pyramid_work[num_scales] = pyramid_orig[num_scales].clone();

    let denoise_sizes: Vec<usize> = pyramid_orig.iter().map(|arr| arr.ncols()).collect();

    let psd_shapes = generate_psd_shapes::<F>(&denoise_sizes);

    let mut denoised: Option<Array2<F>> = None;

    for scale in (0..=num_scales).rev() {
        let img = pyramid_work[scale].clone();

        // Sum-convolution binning scales values by 2^k at scale k
        let norm_factor = F::from_f64_c(2.0f64.powi(scale as i32));

        if img.nrows() < config.bm3d_config.patch_size
            || img.ncols() < config.bm3d_config.patch_size
        {
            if scale > 0 {
                // For skip path, residual captures accumulated changes from coarser scales
                let residual = &img - &pyramid_orig[scale];

                // Scale correction: residual is in 2^scale space, target is in 2^(scale-1) space
                let scale_correction = F::from_f64_c(BINNING_FACTOR as f64);
                let residual_corrected = residual.mapv(|x| x / scale_correction);

                let target_width = pyramid_orig[scale - 1].ncols();
                let debinned_residual = debin_horizontal(
                    residual_corrected.view(),
                    target_width,
                    BINNING_FACTOR,
                    config.debin_iterations,
                );

                let target_rows = pyramid_orig[scale - 1].nrows();
                let mut adjusted_residual = Array2::zeros((target_rows, target_width));
                let copy_rows = debinned_residual.nrows().min(target_rows);
                let copy_cols = debinned_residual.ncols().min(target_width);
                adjusted_residual
                    .slice_mut(s![..copy_rows, ..copy_cols])
                    .assign(&debinned_residual.slice(s![..copy_rows, ..copy_cols]));

                pyramid_work[scale - 1] = &pyramid_orig[scale - 1] + &adjusted_residual;
            }

            denoised = Some(img);
            continue;
        }

        let mut scale_config = config.bm3d_config.clone();
        scale_config.threshold = config.threshold * config.filter_strength;

        let scaled_window = (config.bm3d_config.search_window >> scale).max(8);
        scale_config.search_window = scaled_window;

        let img_normalized = img.mapv(|x| x / norm_factor);

        let estimated_sigma = estimate_noise_sigma_robust(img_normalized.view());

        let dust = F::from_f64_c(1.0e-9);
        if estimated_sigma > dust {
            scale_config.sigma_random = estimated_sigma;
        }

        let den_normalized = bm3d_denoise_segmented(
            img_normalized.view(),
            &scale_config,
            plans,
            psd_shapes.get(scale).map(|p| p.view()),
            DENOISE_SEGMENTS,
            config.bm3d_config.search_window * MIN_DENOISE_SEGMENT_WIDTH_FACTOR,
        )?;

        let den = den_normalized.mapv(|x| x * norm_factor);

        if scale > 0 {
            let residual = &den - &pyramid_orig[scale];

            let (res_rows, res_cols) = residual.dim();
            let n_rows_f = F::from_f64_c(res_rows as f64);
            let mut v_profile = Vec::with_capacity(res_cols);

            let bin_size = 1 << scale;

            for c in 0..res_cols {
                let col = residual.column(c);

                let mut col_data: Vec<F> = col.to_vec();
                let mid = col_data.len() / 2;
                col_data.select_nth_unstable_by(mid, |a, b| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                });
                let med = col_data[mid];
                let med_sq = med * med;

                let mut sum_ac_sq = F::zero();
                for &val in col {
                    let ac = val - med;
                    sum_ac_sq += ac * ac;
                }
                let var_ac = sum_ac_sq / n_rows_f;

                let mut score = if med_sq + var_ac > F::zero() {
                    med_sq / (med_sq + var_ac)
                } else {
                    F::zero()
                };

                let start_idx = c * bin_size;
                let end_idx = (start_idx + bin_size).min(cols);
                let mut max_fine_var = F::zero();
                for &val in &fine_col_vars[start_idx..end_idx] {
                    if val > max_fine_var {
                        max_fine_var = val;
                    }
                }

                if max_fine_var > noise_var_threshold {
                    score = F::zero();
                }

                let s2 = score * score;
                let s4 = s2 * s2;
                let gate = s4 * s4;
                v_profile.push(med * gate);
            }

            let smooth_hump = compute_1d_median_filter(&v_profile, 201);
            let mut spiky_profile = Array2::<F>::zeros((res_rows, res_cols));
            for c in 0..res_cols {
                let spike = v_profile[c] - smooth_hump[c];
                for r in 0..res_rows {
                    spiky_profile[[r, c]] = spike;
                }
            }

            let residual_filtered = spiky_profile;

            let scale_correction = F::from_f64_c(BINNING_FACTOR as f64);
            let residual_corrected = residual_filtered.mapv(|x| x / scale_correction);

            let target_width = pyramid_orig[scale - 1].ncols();
            let debinned_residual = debin_horizontal(
                residual_corrected.view(),
                target_width,
                BINNING_FACTOR,
                config.debin_iterations,
            );

            let target_rows = pyramid_orig[scale - 1].nrows();
            let mut adjusted_residual = Array2::zeros((target_rows, target_width));
            let copy_rows = debinned_residual.nrows().min(target_rows);
            let copy_cols = debinned_residual.ncols().min(target_width);
            adjusted_residual
                .slice_mut(s![..copy_rows, ..copy_cols])
                .assign(&debinned_residual.slice(s![..copy_rows, ..copy_cols]));

            pyramid_work[scale - 1] = &pyramid_orig[scale - 1] + &adjusted_residual;
        }

        denoised = Some(den);
    }

    let final_denoised = denoised.ok_or("No scales processed")?;

    let total_diff = &final_denoised - &sinogram;
    let mut total_profile = Vec::with_capacity(cols);

    for c in 0..cols {
        let mut col_data: Vec<F> = total_diff.column(c).to_vec();
        let mid = col_data.len() / 2;
        col_data.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        total_profile.push(col_data[mid]);
    }

    let final_hump = compute_1d_median_filter(&total_profile, 201);

    let mag_threshold_wall = sigma_global * F::from_f64_c(8.0);

    let silence_threshold = sigma_global * F::from_f64_c(0.5);

    let mut diff_profile = Vec::with_capacity(cols);
    for c in 0..cols {
        diff_profile.push(total_profile[c] - final_hump[c]);
    }

    let mut c = 0;
    while c < cols {
        if diff_profile[c].abs() > silence_threshold {
            let start = c;
            let mut end = c;
            let mut peak_mag = F::zero();

            while end < cols && diff_profile[end].abs() > silence_threshold {
                let mag = diff_profile[end].abs();
                if mag > peak_mag {
                    peak_mag = mag;
                }
                end += 1;
            }
            let width = end - start;

            if width > 20 && peak_mag > mag_threshold_wall {
                diff_profile[start..end].fill(F::zero());
            }

            c = end;
        } else {
            c += 1;
        }
    }

    let mut final_output = sinogram.to_owned();

    for c in 0..cols {
        let streak_val = diff_profile[c];
        for r in 0..rows {
            final_output[[r, c]] += streak_val;
        }
    }

    Ok(final_output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

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
            let u = self.next_u64();
            (u >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    fn random_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
        let mut rng = SimpleLcg::new(seed);
        Array2::from_shape_fn((rows, cols), |_| rng.next_f32())
    }

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_compute_num_scales_small() {
        assert_eq!(compute_num_scales(40), 0);
        assert_eq!(compute_num_scales(30), 0);
        assert_eq!(compute_num_scales(1), 0);
    }

    #[test]
    fn test_compute_num_scales_medium() {
        // Width 80: floor(log2(80/40)) = floor(log2(2)) = 1
        assert_eq!(compute_num_scales(80), 1);
        // Width 160: floor(log2(160/40)) = floor(log2(4)) = 2
        assert_eq!(compute_num_scales(160), 2);
    }

    #[test]
    fn test_compute_num_scales_large() {
        // Width 640: floor(log2(640/40)) = 4, but capped at 3
        assert_eq!(compute_num_scales(640), 3);
        // Width 1280: floor(log2(1280/40)) = 5, but capped at 3
        assert_eq!(compute_num_scales(1280), 3);
    }

    #[test]
    fn test_binning_reduces_width() {
        let arr = random_matrix(32, 100, 12345);
        let binned = bin_horizontal(arr.view(), 2);

        assert!(binned.ncols() <= 51); // 100/2 + 1 for padding edge cases
        assert!(binned.ncols() >= 49);
        assert_eq!(binned.nrows(), 32);
    }

    #[test]
    fn test_binning_factor_1_identity() {
        let arr = random_matrix(32, 64, 11111);
        let binned = bin_horizontal(arr.view(), 1);

        assert_eq!(binned.dim(), arr.dim());
        for (a, b) in arr.iter().zip(binned.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_binning_preserves_sum_approximately() {
        let arr = random_matrix(16, 64, 22222);
        let sum_orig: f32 = arr.iter().sum();

        let binned = bin_horizontal(arr.view(), 2);
        let sum_binned: f32 = binned.iter().sum();

        assert!(
            (sum_binned - sum_orig).abs() < sum_orig * 0.5,
            "Sum changed too much: {} vs {}",
            sum_binned,
            sum_orig
        );
    }

    #[test]
    fn test_debin_restores_width() {
        let original = random_matrix(32, 100, 33333);
        let binned = bin_horizontal(original.view(), 2);

        let debinned = debin_horizontal(binned.view(), 100, 2, 30);

        assert_eq!(debinned.dim(), (32, 100));
    }

    #[test]
    fn test_bin_debin_approximate_recovery() {
        let original = random_matrix(16, 64, 44444);
        let binned = bin_horizontal(original.view(), 2);
        let debinned = debin_horizontal(binned.view(), 64, 2, 30);

        let mae: f32 = original
            .iter()
            .zip(debinned.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / (original.len() as f32);

        let mean_val: f32 = original.iter().sum::<f32>() / (original.len() as f32);
        assert!(
            mae < mean_val * 0.5,
            "MAE {} too large relative to mean {}",
            mae,
            mean_val
        );
    }

    #[test]
    fn test_debin_factor_1_identity() {
        let arr = random_matrix(16, 32, 55555);
        let debinned = debin_horizontal(arr.view(), 32, 1, 30);

        assert_eq!(debinned.dim(), arr.dim());
    }

    #[test]
    fn test_psd_shapes_count() {
        let sizes = vec![256, 128, 64, 32];
        let shapes = generate_psd_shapes::<f32>(&sizes);

        assert_eq!(shapes.len(), 4);
    }

    #[test]
    fn test_psd_shapes_sizes() {
        let sizes = vec![256, 128, 64, 32];
        let shapes = generate_psd_shapes::<f32>(&sizes);

        for (i, shape) in shapes.iter().enumerate() {
            assert_eq!(shape.len(), sizes[i], "Shape {} has wrong size", i);
        }
    }

    #[test]
    fn test_psd_coarsest_flat() {
        let sizes = vec![128, 64, 32];
        let shapes = generate_psd_shapes::<f32>(&sizes);

        let coarsest = &shapes[shapes.len() - 1];
        let first_val = coarsest[0];

        for &val in coarsest.iter() {
            assert!(
                approx_eq(val, first_val, 1e-6),
                "Coarsest PSD should be flat"
            );
        }
    }

    #[test]
    fn test_psd_all_non_negative() {
        let sizes = vec![256, 128, 64];
        let shapes = generate_psd_shapes::<f32>(&sizes);

        for (i, shape) in shapes.iter().enumerate() {
            for &val in shape.iter() {
                assert!(val >= 0.0, "PSD shape {} has negative value", i);
            }
        }
    }

    #[test]
    fn test_default_config() {
        let config: MultiscaleConfig<f32> = MultiscaleConfig::default();

        assert!(config.num_scales.is_none());
        assert!(approx_eq(config.filter_strength, 1.0, 1e-6));
        assert!(approx_eq(config.threshold, 3.5, 1e-6));
        assert_eq!(config.debin_iterations, 30);
    }

    #[test]
    fn test_config_validation_valid() {
        let config: MultiscaleConfig<f32> = MultiscaleConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_invalid_filter_strength() {
        let config: MultiscaleConfig<f32> = MultiscaleConfig {
            filter_strength: 0.0,
            ..MultiscaleConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_debin_iterations() {
        let config: MultiscaleConfig<f32> = MultiscaleConfig {
            debin_iterations: 0,
            ..MultiscaleConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_multiscale_smoke_small() {
        // Small image that will use single-scale (K=0)
        let image = random_matrix(32, 32, 66666);
        let config = MultiscaleConfig::default();

        let result = multiscale_bm3d_streak_removal(image.view(), &config);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.dim(), image.dim());
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_multiscale_smoke_medium() {
        // Medium image with K > 0
        let image = random_matrix(64, 128, 77777);
        let config = MultiscaleConfig::default();

        let result = multiscale_bm3d_streak_removal(image.view(), &config);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.dim(), image.dim());
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn non_positive_input_blank_passes_varying_errors() {
        let cfg = MultiscaleConfig::<f32>::default();

        let blank = Array2::<f32>::zeros((64, 128));
        let out = multiscale_bm3d_streak_removal(blank.view(), &cfg).unwrap();
        assert_eq!(out, blank, "a blank slice must pass through unchanged");

        // Varying but nowhere positive: the shape of accidentally pre-logged
        // transmission data (ln of (0,1) data is negative everywhere).
        let logged_by_mistake = random_matrix(64, 128, 777).mapv(|v| (v * 0.5 + 0.25).ln());
        assert!(logged_by_mistake.iter().all(|&v| v <= 0.0));
        let err = multiscale_bm3d_streak_removal(logged_by_mistake.view(), &cfg)
            .expect_err("varying non-positive input must be rejected");
        assert!(
            err.contains("log_domain_input"),
            "the error must name the opt-in flag, got: {err}"
        );

        let flagged = MultiscaleConfig::<f32> {
            log_domain_input: true,
            ..MultiscaleConfig::default()
        };
        multiscale_bm3d_streak_removal(logged_by_mistake.view(), &flagged)
            .expect("declared log-domain input must be accepted");
    }

    #[test]
    fn log_domain_flag_is_a_pure_change_of_domain() {
        let image = random_matrix(48, 200, 31415).mapv(|v| v + 0.05); // strictly positive
        let default_cfg = MultiscaleConfig::<f32>::default();
        let flagged_cfg = MultiscaleConfig::<f32> {
            log_domain_input: true,
            ..MultiscaleConfig::default()
        };

        let via_default = multiscale_bm3d_streak_removal(image.view(), &default_cfg).unwrap();

        let logged = image.mapv(f32::ln);
        let via_flag = multiscale_bm3d_streak_removal(logged.view(), &flagged_cfg)
            .unwrap()
            .mapv(f32::exp);

        for (a, b) in via_default.iter().zip(via_flag.iter()) {
            assert!(
                approx_eq(*a, *b, 1e-4),
                "log wrapper is not a pure change of domain: {} vs {}",
                a,
                b
            );
        }
    }

    fn single_scale_in_log_domain(image: &Array2<f32>, config: &Bm3dConfig<f32>) -> Array2<f32> {
        let floor = image
            .iter()
            .copied()
            .filter(|&v| v > 0.0)
            .fold(f32::INFINITY, f32::min);
        let logged = image.mapv(|v| if v > 0.0 { v.ln() } else { floor.ln() });
        let denoised =
            bm3d_ring_artifact_removal(logged.view(), RingRemovalMode::Streak, config).unwrap();
        denoised.mapv(f32::exp)
    }

    #[test]
    fn test_k0_equals_single_scale() {
        // For small images (K=0), multiscale reduces to single-scale streak
        // removal applied in the log domain (the multiscale entry logs its input).
        let image = random_matrix(32, 32, 88888);
        let config = MultiscaleConfig::default();

        let multiscale_result = multiscale_bm3d_streak_removal(image.view(), &config).unwrap();
        let single_result = single_scale_in_log_domain(&image, &config.bm3d_config);

        for (a, b) in multiscale_result.iter().zip(single_result.iter()) {
            assert!(
                approx_eq(*a, *b, 1e-5),
                "K=0 results differ: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_multiscale_differs_from_single() {
        let image = random_matrix(64, 256, 99999);

        let single_config = Bm3dConfig {
            threshold: 3.5,
            ..Bm3dConfig::default()
        };

        let single_result =
            bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Streak, &single_config)
                .unwrap();

        let multi_config = MultiscaleConfig::default();
        let multi_result = multiscale_bm3d_streak_removal(image.view(), &multi_config).unwrap();

        let diff: f32 = single_result
            .iter()
            .zip(multi_result.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(
            diff > 0.01,
            "Multiscale should produce different results than single-scale"
        );
    }

    #[test]
    fn test_multiscale_handles_wide_streak() {
        let mut image = Array2::from_elem((64, 256), 0.5f32);
        for r in 0..64 {
            for c in 60..64 {
                image[[r, c]] = 0.9;
            }
        }

        let config = MultiscaleConfig {
            bm3d_config: Bm3dConfig {
                sigma_random: 0.5,
                ..Bm3dConfig::default()
            },
            ..MultiscaleConfig::default()
        };
        let result = multiscale_bm3d_streak_removal(image.view(), &config);

        assert!(result.is_ok());
        let output = result.unwrap();

        let col_means: Vec<f32> = (0..256)
            .map(|c| {
                let sum: f32 = (0..64).map(|r| output[[r, c]]).sum();
                sum / 64.0
            })
            .collect();

        let overall_mean: f32 = col_means.iter().sum::<f32>() / 256.0;
        let col_variance: f32 = col_means
            .iter()
            .map(|m| (m - overall_mean).powi(2))
            .sum::<f32>()
            / 256.0;

        let orig_col_means: Vec<f32> = (0..256)
            .map(|c| {
                let sum: f32 = (0..64).map(|r| image[[r, c]]).sum();
                sum / 64.0
            })
            .collect();
        let orig_overall_mean: f32 = orig_col_means.iter().sum::<f32>() / 256.0;
        let orig_col_variance: f32 = orig_col_means
            .iter()
            .map(|m| (m - orig_overall_mean).powi(2))
            .sum::<f32>()
            / 256.0;

        assert!(
            col_variance < orig_col_variance,
            "Multiscale should reduce wide streak variance: {} >= {}",
            col_variance,
            orig_col_variance
        );
    }

    #[test]
    fn test_multiscale_f64() {
        let image = Array2::from_shape_fn((32, 128), |(r, c)| (r * 128 + c) as f64 / 4096.0);
        let config: MultiscaleConfig<f64> = MultiscaleConfig::default();

        let result = multiscale_bm3d_streak_removal(image.view(), &config);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.dim(), image.dim());
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_override_num_scales() {
        let image = random_matrix(64, 256, 10101);

        let config = MultiscaleConfig {
            num_scales: Some(1),
            ..MultiscaleConfig::default()
        };

        let result = multiscale_bm3d_streak_removal(image.view(), &config);

        assert!(result.is_ok());
    }

    #[test]
    fn test_force_k0() {
        let image = random_matrix(64, 256, 20202);

        let config = MultiscaleConfig {
            num_scales: Some(0),
            ..MultiscaleConfig::default()
        };

        let multi_result = multiscale_bm3d_streak_removal(image.view(), &config).unwrap();
        let single_result = single_scale_in_log_domain(&image, &config.bm3d_config);

        for (a, b) in multi_result.iter().zip(single_result.iter()) {
            assert!(approx_eq(*a, *b, 1e-5), "Forced K=0 differs from single");
        }
    }

    fn flat_air_streaked(rows: usize, cols: usize, air: usize) -> Array2<f32> {
        let mut rng = SimpleLcg::new(9);
        let jitter: Vec<f32> = (0..cols).map(|_| (rng.next_f32() - 0.5) * 0.04).collect();
        Array2::from_shape_fn((rows, cols), |(r, c)| {
            if c < air || c >= cols - air {
                1.0e-8 * (1.0 + jitter[c])
            } else {
                let row_term = 1.0 + 0.1 * (r as f32 / rows as f32);
                0.5 * row_term * (1.0 + jitter[c])
            }
        })
    }

    #[test]
    fn vertical_binning_averages_exactly() {
        // 7 rows, factor 3: bins of 3, 3, and a ragged 1.
        let img = Array2::from_shape_fn((7, 4), |(r, c)| (r * 10 + c) as f32);
        let binned = bin_vertical_mean(img.view(), 3);
        assert_eq!(binned.dim(), (3, 4));
        for c in 0..4 {
            let c_f = c as f32;
            assert!(approx_eq(binned[[0, c]], 10.0 + c_f, 1e-6)); // rows 0,1,2
            assert!(approx_eq(binned[[1, c]], 40.0 + c_f, 1e-6)); // rows 3,4,5
            assert!(approx_eq(binned[[2, c]], 60.0 + c_f, 1e-6)); // row 6 alone
        }
        let same = bin_vertical_mean(img.view(), 1);
        assert_eq!(same, img);
    }

    #[test]
    fn segmented_call_falls_back_to_plain_below_minimum_width() {
        let image = random_matrix(48, 90, 4242).mapv(|v| v + 0.05);
        let config = Bm3dConfig::<f32>::default();
        let plans = crate::pipeline::Bm3dPlans::new(config.patch_size, config.max_matches);

        // min width 96 > 90 cols: no segmentation possible.
        let segmented = bm3d_denoise_segmented(image.view(), &config, &plans, None, 8, 96).unwrap();
        let plain = crate::orchestration::bm3d_ring_artifact_removal_with_psd(
            image.view(),
            RingRemovalMode::Streak,
            &config,
            &plans,
            None,
        )
        .unwrap();
        assert_eq!(
            segmented, plain,
            "narrow-level fallback must be identical to the unsegmented call"
        );
    }

    fn tapered_streaked(rows: usize, cols: usize, air: usize, taper: usize) -> Array2<f32> {
        let mut rng = SimpleLcg::new(9);
        let jitter: Vec<f32> = (0..cols).map(|_| (rng.next_f32() - 0.5) * 0.04).collect();
        Array2::from_shape_fn((rows, cols), |(r, c)| {
            let envelope = if c + taper < air || c >= cols - air + taper {
                0.0
            } else if c < air {
                let x = (c + taper - air) as f32 / taper as f32;
                0.5 * (1.0 - (std::f32::consts::PI * x).cos())
            } else if c + air + taper >= cols + taper && c + air >= cols {
                0.0
            } else if c + air + taper >= cols {
                let x = (c + air + taper - cols) as f32 / taper as f32;
                0.5 * (1.0 + (std::f32::consts::PI * x).cos())
            } else {
                1.0
            };
            let row_term = 1.0 + 0.1 * (r as f32 / rows as f32);
            (1.0e-8 + envelope * 0.5 * row_term) * (1.0 + jitter[c])
        })
    }

    #[test]
    fn tall_input_binned_path_corrects_streaks_and_preserves_rows() {
        let rows = VERTICAL_BIN_TARGET_ROWS * 2; // guarantees factor >= 2
        let img = tapered_streaked(rows, 200, 60, 30);
        let cfg = MultiscaleConfig::<f32> {
            num_scales: Some(2),
            ..Default::default()
        };
        let out = multiscale_bm3d_streak_removal(img.view(), &cfg).unwrap();

        let max_change = img
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_change > 1.0e-4,
            "binned path returned its input unchanged: max|change| = {max_change}"
        );

        for c in [70usize, 100, 130] {
            let mut deviation = 0.0f32;
            let mut mean = 0.0f32;
            for r in 0..rows {
                mean += (out[[r, c]] / img[[r, c]]).ln();
            }
            mean /= rows as f32;
            for r in 0..rows {
                deviation = deviation.max(((out[[r, c]] / img[[r, c]]).ln() - mean).abs());
            }
            assert!(
                deviation < 1.0e-3,
                "column {c}: correction is not a per-column offset (max dev {deviation})"
            );
        }
    }

    #[test]
    fn auto_sigma_skips_numerically_silent_patches() {
        let img = flat_air_streaked(128, 256, 80);
        let sigma = estimate_noise_sigma_robust(img.view());
        let lo = img.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = img.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            f64::from(sigma) > 1.0e-6 * f64::from(hi - lo),
            "estimate is quantization residue, not a noise floor: {sigma}"
        );
    }

    #[test]
    fn auto_sigma_tracks_the_streak_amplitude() {
        let (rows, cols) = (128usize, 256usize);
        let amplitude = 0.05f32;
        let mut img = Array2::<f32>::zeros((rows, cols));
        for c in 0..cols {
            let offset = if c % 2 == 0 { amplitude } else { -amplitude };
            for r in 0..rows {
                // A smooth object so the estimator has signal to reject.
                let object = 0.5 + 0.2 * ((r as f32) / (rows as f32)).sin();
                img[[r, c]] = object + offset;
            }
        }

        let sigma = f64::from(estimate_noise_sigma_robust(img.view()));
        let truth = f64::from(amplitude);
        assert!(
            sigma > truth / 3.0 && sigma < truth * 3.0,
            "auto sigma {sigma:.3e} is not within 3x of the streak amplitude {truth:.3e}"
        );
    }

    #[test]
    fn auto_sigma_ignores_empty_margins() {
        let (rows, cols, margin) = (128usize, 256usize, 80usize);
        let amplitude = 0.05f32;
        let mut with_air = Array2::<f32>::zeros((rows, cols));
        for c in 0..cols {
            for r in 0..rows {
                with_air[[r, c]] = 1.0e-8 + 1.0e-9 * ((r + c) % 3) as f32;
            }
        }
        for c in margin..(cols - margin) {
            let offset = if c % 2 == 0 { amplitude } else { -amplitude };
            for r in 0..rows {
                with_air[[r, c]] = 0.5 + offset;
            }
        }

        let sigma = f64::from(estimate_noise_sigma_robust(with_air.view()));
        let truth = f64::from(amplitude);
        assert!(
            sigma > truth / 3.0,
            "empty margins dragged the estimate to {sigma:.3e}, far under the \
             streak amplitude {truth:.3e} present in the object"
        );
    }

    #[test]
    fn multiscale_auto_acts_on_flat_background() {
        let img = flat_air_streaked(96, 200, 60);
        let mut cfg = MultiscaleConfig::<f32> {
            num_scales: Some(2),
            ..Default::default()
        };
        cfg.bm3d_config.sigma_random = 0.0; // auto-estimate

        let out = multiscale_bm3d_streak_removal(img.view(), &cfg).unwrap();

        let max_change = img
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_change > 1.0e-4,
            "multiscale auto returned its input unchanged: max|change| = {max_change}"
        );
    }

    #[test]
    fn fully_flat_input_stays_a_noop() {
        let img = Array2::<f32>::from_elem((96, 200), 0.75);
        let sigma = estimate_noise_sigma_robust(img.view());
        assert!(
            f64::from(sigma) < 1.0e-9,
            "flat input produced a nonzero noise floor: {sigma}"
        );
    }
}

pub fn compute_residual_kernel<F: Bm3dFloat>() -> Vec<f64> {
    let kernel0 = compute_residual_kernel_shift::<F>(0);
    let kernel1 = compute_residual_kernel_shift::<F>(1);

    kernel0
        .iter()
        .zip(kernel1.iter())
        .map(|(a, b)| (a + b) / 2.0)
        .collect()
}

fn compute_residual_kernel_shift<F: Bm3dFloat>(shift: usize) -> Vec<f64> {
    let factor = BINNING_FACTOR;
    let iterations = DEFAULT_DEBIN_ITERATIONS;

    // Use a large enough width to avoid boundary effects
    let width = 512;
    let center = width / 2 + shift;

    let mut impulse = Array2::<F>::zeros((1, width));
    impulse[[0, center]] = F::one();

    let binned = bin_horizontal(impulse.view(), factor);

    let scale_correction = F::from_f64_c(factor as f64);
    let binned_normalized = binned.mapv(|x| x / scale_correction);

    let rebuilt = debin_horizontal(binned_normalized.view(), width, factor, iterations);

    let mut residual_kernel = Vec::new();
    let kernel_radius = 18;

    for i in (center - kernel_radius)..=(center) {
        let val = impulse[[0, i]] - rebuilt[[0, i]];
        residual_kernel.push(val.to_f64().unwrap());
    }

    residual_kernel
}

#[cfg(test)]
mod kernel_tests {
    use super::*;

    #[test]
    fn test_verify_residual_kernel_properties() {
        let computed = compute_residual_kernel::<f64>();

        let center_idx = 18;
        let center_val = computed[center_idx];
        assert!(
            center_val > 0.7,
            "Center value should be significant (residual of peak)"
        );

        let mut full_kernel = computed.clone();
        for i in (0..center_idx).rev() {
            full_kernel.push(computed[i]);
        }
        let sum: f64 = full_kernel.iter().sum();
        assert!(
            (sum - 0.5).abs() < 0.1,
            "Residual kernel sum reflects pipeline DC damping (0.5)"
        );
    }
}
