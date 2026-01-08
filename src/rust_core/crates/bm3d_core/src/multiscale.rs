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
//!
//! This enables handling of wide streaks that single-scale BM3D cannot capture.

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use splines::{Interpolation, Key, Spline};

use crate::float_trait::Bm3dFloat;
use crate::orchestration::{bm3d_ring_artifact_removal, Bm3dConfig, RingRemovalMode};

// =============================================================================
// Constants
// =============================================================================

/// Minimum width before stopping binning (from reference: 40 pixels)
const MIN_SCALE_WIDTH: usize = 40;

/// Default binning factor (always halves)
const BINNING_FACTOR: usize = 2;

/// Default number of debinning iterations
const DEFAULT_DEBIN_ITERATIONS: usize = 30;

/// Default filter strength for multi-scale (reference uses 1.0)
const DEFAULT_MULTISCALE_FILTER_STRENGTH: f64 = 1.0;

/// Default threshold for multi-scale (reference uses 3.5, different from single-scale 2.7)
const DEFAULT_MULTISCALE_THRESHOLD: f64 = 3.5;

/// Pre-computed residual kernel half (from bm3d-streak-removal _internal.py:82-84)
/// This is the noise transfer function for the binning/debinning process
const RESIDUAL_KERNEL_HALF: [f64; 19] = [
    -0.000038014,
    -0.00018945,
    -0.0005779,
    -0.0006854,
    -0.00017558,
    -0.00078497,
    -0.002597,
    -0.0017638,
    0.0014121,
    -0.0012586,
    -0.0087586,
    -0.0045105,
    0.0062736,
    -0.0058483,
    -0.021518,
    -0.010869,
    -0.060379,
    -0.2232,
    0.68939,
];

// =============================================================================
// Types
// =============================================================================

/// Configuration for multi-scale BM3D processing.
#[derive(Debug, Clone)]
pub struct MultiscaleConfig<F: Bm3dFloat> {
    /// Override automatic scale calculation; if None, use floor(log2(width/40))
    pub num_scales: Option<usize>,
    /// Multiplier for BM3D filtering intensity. Default: 1.0
    pub filter_strength: F,
    /// Hard threshold coefficient. Default: 3.5 (note: different from single-scale 2.7)
    pub threshold: F,
    /// Iterations for cubic spline debinning. Default: 30
    pub debin_iterations: usize,
    /// Inner BM3D config (patch_size, search_window, etc.)
    pub bm3d_config: Bm3dConfig<F>,
}

impl<F: Bm3dFloat> Default for MultiscaleConfig<F> {
    fn default() -> Self {
        let mut bm3d_config = Bm3dConfig::default();
        // Override threshold to multi-scale default
        bm3d_config.threshold = F::from_f64_c(DEFAULT_MULTISCALE_THRESHOLD);
        bm3d_config.filter_strength = F::from_f64_c(DEFAULT_MULTISCALE_FILTER_STRENGTH);

        Self {
            num_scales: None,
            filter_strength: F::from_f64_c(DEFAULT_MULTISCALE_FILTER_STRENGTH),
            threshold: F::from_f64_c(DEFAULT_MULTISCALE_THRESHOLD),
            debin_iterations: DEFAULT_DEBIN_ITERATIONS,
            bm3d_config,
        }
    }
}

impl<F: Bm3dFloat> MultiscaleConfig<F> {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate the configuration parameters.
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

// =============================================================================
// Helper Functions: Binning
// =============================================================================

/// Compute the number of horizontal scales based on image width.
/// K = max(0, floor(log2(width / MIN_SCALE_WIDTH)))
pub fn compute_num_scales(width: usize) -> usize {
    if width <= MIN_SCALE_WIDTH {
        return 0;
    }
    let ratio = width as f64 / MIN_SCALE_WIDTH as f64;
    ratio.log2().floor().max(0.0) as usize
}

/// Symmetric padding on the right side of a 2D array (horizontal dimension).
fn symmetric_pad_right<F: Bm3dFloat>(arr: ArrayView2<F>, pad_amount: usize) -> Array2<F> {
    if pad_amount == 0 {
        return arr.to_owned();
    }

    let (rows, cols) = arr.dim();
    let new_cols = cols + pad_amount;
    let mut result = Array2::zeros((rows, new_cols));

    // Copy original data
    result.slice_mut(s![.., ..cols]).assign(&arr);

    // Mirror padding: reflect from the right edge
    for p in 0..pad_amount {
        let src_col = cols - 1 - (p % cols);
        for r in 0..rows {
            result[[r, cols + p]] = arr[[r, src_col]];
        }
    }

    result
}

/// Bin a 2D array horizontally by factor h using sum-convolution.
/// Follows the exact algorithm from bm3d-streak-removal _internal.py:339-360
pub fn bin_horizontal<F: Bm3dFloat>(arr: ArrayView2<F>, factor: usize) -> Array2<F> {
    if factor <= 1 {
        return arr.to_owned();
    }

    let (rows, cols) = arr.dim();

    // Step 1: Symmetric padding to make width align for binning
    let mod_pad = factor - ((cols - 1) % factor) - 1;
    let padded = symmetric_pad_right(arr, mod_pad);
    let padded_cols = padded.ncols();

    // Step 2: Convolve horizontally with kernel of ones (sum over factor columns)
    let mut convolved = Array2::zeros((rows, padded_cols));
    for r in 0..rows {
        for c in 0..padded_cols {
            let mut sum = F::zero();
            for k in 0..factor {
                let src_c = c as isize + k as isize - (factor as isize / 2);
                if src_c >= 0 && (src_c as usize) < padded_cols {
                    sum = sum + padded[[r, src_c as usize]];
                }
            }
            convolved[[r, c]] = sum;
        }
    }

    // Step 3: Sample at bin centers
    let h_half = factor / 2;
    let start = h_half + if factor % 2 == 1 { 1 } else { 0 };
    let end = padded_cols - h_half + 1;

    let num_bins = (end - start + factor - 1) / factor;
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

/// Compute binning weights for debinning normalization.
/// This matches the n_counter computation in _internal.py:383
fn compute_bin_weights<F: Bm3dFloat>(target_width: usize, factor: usize) -> Array1<F> {
    if factor <= 1 {
        return Array1::ones(target_width);
    }

    // Create a row of ones and apply the same binning convolution
    let ones = Array2::from_elem((1, target_width), F::one());
    let mod_pad = factor - ((target_width - 1) % factor) - 1;
    let padded = symmetric_pad_right(ones.view(), mod_pad);
    let padded_cols = padded.ncols();

    // Convolve with ones kernel
    let mut convolved = Array1::zeros(padded_cols);
    for c in 0..padded_cols {
        let mut sum = F::zero();
        for k in 0..factor {
            let src_c = c as isize + k as isize - (factor as isize / 2);
            if src_c >= 0 && (src_c as usize) < padded_cols {
                sum = sum + padded[[0, src_c as usize]];
            }
        }
        convolved[c] = sum;
    }

    // Return the relevant portion for the target width
    convolved.slice(s![..target_width]).to_owned()
}

// =============================================================================
// Helper Functions: Debinning (Cubic Spline Interpolation)
// =============================================================================

/// Cubic spline interpolation for a single row.
/// Uses Catmull-Rom spline which approximates natural cubic spline behavior.
fn cubic_spline_interpolate_row<F: Bm3dFloat>(
    x_coords: &[f64],
    y_values: ArrayView1<F>,
    target_x: &[f64],
) -> Vec<F> {
    // Build spline keys
    // Note: to_f64() from num_traits::Float returns Option<f64>, but f32/f64 always convert successfully
    let keys: Vec<Key<f64, f64>> = x_coords
        .iter()
        .zip(y_values.iter())
        .map(|(&x, &y)| {
            let y_f64 = y.to_f64().unwrap_or(0.0);
            Key::new(x, y_f64, Interpolation::CatmullRom)
        })
        .collect();

    let spline = Spline::from_vec(keys);

    // Sample at target positions
    target_x
        .iter()
        .map(|&x| {
            // Handle extrapolation at boundaries
            let clamped_x = x.clamp(x_coords[0], x_coords[x_coords.len() - 1]);
            let val = spline.sample(clamped_x).unwrap_or(0.0);
            F::from_f64_c(val)
        })
        .collect()
}

/// Debin a 2D array horizontally using iterative cubic spline interpolation.
/// Follows the algorithm from bm3d-streak-removal _internal.py:363-408
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

    // Compute bin weights for normalization
    let n_counter = compute_bin_weights::<F>(target_width, factor);

    // Coordinates of bin centers in target space
    // x1c: indices where bins are located
    // x1: actual x coordinates for interpolation
    let x1c: Vec<usize> = (0..binned_cols)
        .map(|i| h_half + (if factor % 2 == 1 { 1 } else { 0 }) + i * factor)
        .filter(|&c| c < target_width)
        .collect();

    let x1: Vec<f64> = (0..binned_cols)
        .map(|i| {
            (h_half as f64)
                + 1.0
                - (if factor % 2 == 0 { 0.5 } else { 0.0 })
                + (i * factor) as f64
        })
        .collect();

    // Target coordinates (1..target_width)
    let ix1: Vec<f64> = (1..=target_width).map(|i| i as f64).collect();

    // Iterative refinement
    let mut y_j = Array2::zeros((rows, target_width));

    for iter in 0..iterations.max(1) {
        // Compute residual
        let r_j = if iter > 0 {
            let binned_y_j = bin_horizontal(y_j.view(), factor);
            // Ensure dimensions match for subtraction
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

        // Normalize by bin weights at x1c positions
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

        // Cubic spline interpolation for each row
        let x1_trimmed: Vec<f64> = x1.iter().take(normalized.ncols()).copied().collect();
        if x1_trimmed.len() >= 2 {
            for r in 0..rows {
                let row_values = normalized.row(r);
                let row_slice = row_values.slice(s![..x1_trimmed.len()]);
                let interpolated = cubic_spline_interpolate_row(&x1_trimmed, row_slice, &ix1);

                for (c, &val) in interpolated.iter().enumerate() {
                    if c < target_width {
                        y_j[[r, c]] = y_j[[r, c]] + val;
                    }
                }
            }
        }
    }

    y_j
}

// =============================================================================
// Helper Functions: PSD Shape Generation
// =============================================================================

/// Generate PSD shapes for each scale based on the residual kernel.
/// Follows bm3d-streak-removal _internal.py:69-99
pub fn generate_psd_shapes<F: Bm3dFloat>(denoise_sizes: &[usize]) -> Vec<Array1<F>> {
    let mut psd_shapes = Vec::with_capacity(denoise_sizes.len());

    for i in 0..denoise_sizes.len() {
        let sz = denoise_sizes[i];

        if i < denoise_sizes.len() - 1 {
            // Non-coarsest scales: use residual kernel PSD
            let kernel_half_len = RESIDUAL_KERNEL_HALF.len();
            let full_kernel_len = kernel_half_len * 2 - 1;

            // Trim kernel if needed
            let trim_amount = if full_kernel_len > sz {
                (full_kernel_len - sz) / 2
            } else {
                0
            };
            let trimmed_start = trim_amount.min(kernel_half_len - 1);
            let trimmed_half: Vec<f64> = RESIDUAL_KERNEL_HALF[trimmed_start..].to_vec();

            // Build symmetric kernel
            let mut residual_kernel: Vec<f64> = trimmed_half.clone();
            for j in (0..trimmed_half.len() - 1).rev() {
                residual_kernel.push(trimmed_half[j]);
            }

            // Compute FFT magnitude squared (PSD)
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
    use rustfft::{num_complex::Complex, FftPlanner};

    // Pad or truncate kernel to target size
    let mut padded = vec![Complex::new(0.0, 0.0); target_size];
    for (i, &val) in kernel.iter().enumerate() {
        if i < target_size {
            padded[i] = Complex::new(val, 0.0);
        }
    }

    // Compute FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(target_size);
    fft.process(&mut padded);

    // Compute |FFT|²
    Array1::from_iter(padded.iter().map(|c| F::from_f64_c(c.norm_sqr())))
}

// =============================================================================
// MAD-Based Sigma Estimation (Mäkinen et al., Section 3.3.2)
// =============================================================================

/// MAD calibration factor for Gaussian distribution (1 / Φ^{-1}(3/4))
const MAD_CALIBRATION: f64 = 1.4826;

/// Daubechies-3 high-pass filter coefficients (db3 wavelet decomposition high-pass)
/// Used for horizontal high-pass filtering to isolate noise from signal.
const DB3_HIGHPASS: [f64; 6] = [
    0.03522629188210,
    -0.08544127388224,
    -0.13501102001039,
    0.45987750211933,
    -0.80689150931109,
    0.33267055295096,
];

/// Compute median of a slice (modifies the slice by partial sorting).
fn median_of_slice<F: Bm3dFloat>(data: &mut [F]) -> F {
    if data.is_empty() {
        return F::zero();
    }
    let mid = data.len() / 2;
    // Partial sort to find median
    data.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if data.len() % 2 == 1 {
        data[mid]
    } else {
        // For even length, average of two middle elements
        let lower = data[..mid].iter().copied().fold(F::neg_infinity(), |a, b| if b > a { b } else { a });
        (lower + data[mid]) / F::from_f64_c(2.0)
    }
}

/// Compute Median Absolute Deviation (MAD) of an array.
/// MAD = median(|x - median(x)|)
/// Returns calibrated estimate: 1.4826 * MAD (for Gaussian noise, this equals σ)
fn compute_mad<F: Bm3dFloat>(data: ArrayView2<F>) -> F {
    let mut flat: Vec<F> = data.iter().copied().collect();
    if flat.is_empty() {
        return F::zero();
    }

    // Compute median
    let med = median_of_slice(&mut flat);

    // Compute absolute deviations from median
    let mut deviations: Vec<F> = flat.iter().map(|&x| (x - med).abs()).collect();

    // Compute MAD
    let mad = median_of_slice(&mut deviations);

    // Calibrate for Gaussian distribution
    mad * F::from_f64_c(MAD_CALIBRATION)
}

/// Apply 1D convolution to each row of a 2D array.
fn convolve_rows<F: Bm3dFloat>(arr: ArrayView2<F>, kernel: &[f64]) -> Array2<F> {
    let (rows, cols) = arr.dim();
    let klen = kernel.len();
    let half = klen / 2;

    let mut result = Array2::zeros((rows, cols));

    for r in 0..rows {
        for c in 0..cols {
            let mut sum = F::zero();
            for (ki, &kval) in kernel.iter().enumerate() {
                let src_c = c as isize + ki as isize - half as isize;
                if src_c >= 0 && (src_c as usize) < cols {
                    sum = sum + arr[[r, src_c as usize]] * F::from_f64_c(kval);
                }
            }
            result[[r, c]] = sum;
        }
    }

    result
}

/// Apply 1D convolution to each column of a 2D array.
fn convolve_cols<F: Bm3dFloat>(arr: ArrayView2<F>, kernel: &[f64]) -> Array2<F> {
    let (rows, cols) = arr.dim();
    let klen = kernel.len();
    let half = klen / 2;

    let mut result = Array2::zeros((rows, cols));

    for r in 0..rows {
        for c in 0..cols {
            let mut sum = F::zero();
            for (ki, &kval) in kernel.iter().enumerate() {
                let src_r = r as isize + ki as isize - half as isize;
                if src_r >= 0 && (src_r as usize) < rows {
                    sum = sum + arr[[src_r as usize, c]] * F::from_f64_c(kval);
                }
            }
            result[[r, c]] = sum;
        }
    }

    result
}

/// Generate a 1D Gaussian kernel.
fn gaussian_kernel(length: usize, sigma: f64) -> Vec<f64> {
    let center = length as f64 / 2.0;
    let mut kernel: Vec<f64> = (0..length)
        .map(|i| {
            let x = i as f64 - center;
            (-0.5 * (x / sigma).powi(2)).exp()
        })
        .collect();

    // Normalize
    let sum: f64 = kernel.iter().sum();
    if sum > 0.0 {
        for k in &mut kernel {
            *k /= sum;
        }
    }

    kernel
}

/// Estimate noise standard deviation using MAD-based robust estimation.
///
/// This implements the σ estimation from Mäkinen et al. (2021), Section 3.3.2.
/// The image is convolved with a 2D kernel (vertical Gaussian × horizontal high-pass),
/// then MAD is computed and scaled to estimate the noise std.
///
/// The key insight is that:
/// 1. Vertical Gaussian smoothing reduces random noise (streaks are vertically correlated)
/// 2. Horizontal high-pass (db3 wavelet) removes signal, leaving mostly noise
/// 3. MAD is robust to outliers (image features that survive filtering)
///
/// # Arguments
/// * `image` - 2D array (rows × cols), should be the working image Z*_k
///
/// # Returns
/// Estimated noise standard deviation (ς_k in paper notation)
pub fn estimate_sigma_mad<F: Bm3dFloat>(image: ArrayView2<F>) -> F {
    let (rows, _cols) = image.dim();

    // Step 1: Vertical Gaussian smoothing
    // Paper uses length m_v/2, sigma m_v/12 where m_v ≈ 64
    // We adapt based on actual image height
    let gauss_len = (rows / 2).max(5).min(32);
    let gauss_sigma = gauss_len as f64 / 6.0; // Paper uses m_v/12, we use length/6
    let gauss_kernel = gaussian_kernel(gauss_len, gauss_sigma);

    // Apply vertical Gaussian (column-wise convolution)
    let smoothed = convolve_cols(image, &gauss_kernel);

    // Step 2: Horizontal high-pass filtering with db3 wavelet
    let filtered = convolve_rows(smoothed.view(), &DB3_HIGHPASS);

    // Step 3: Compute MAD (robust noise estimate)
    let raw_sigma = compute_mad(filtered.view());

    // Step 4: Normalize by kernel energy
    // The db3 wavelet has L2 norm ≈ 1.0, so minimal correction needed
    // but we account for the combined filter response
    let db3_energy: f64 = DB3_HIGHPASS.iter().map(|x| x * x).sum::<f64>().sqrt();

    raw_sigma / F::from_f64_c(db3_energy)
}

// =============================================================================
// Main Entry Point
// =============================================================================

/// Multi-scale BM3D streak removal for 2D sinograms.
///
/// This function wraps single-scale BM3D in a coarse-to-fine pyramid
/// for handling wide streaks that single-scale cannot capture.
///
/// # Algorithm
///
/// 1. Build horizontal pyramid via 2× binning
/// 2. Start at coarsest scale, denoise with BM3D
/// 3. Compute residual (denoised - original at this scale)
/// 4. Debin residual to finer scale using cubic spline
/// 5. Add debinned residual to finer scale's original
/// 6. Repeat until finest scale
///
/// # Arguments
///
/// * `sinogram` - Input 2D sinogram (H × W), assumes log-transformed data
/// * `config` - Multi-scale configuration parameters
///
/// # Returns
///
/// Denoised sinogram with same shape as input.
///
/// # Example
///
/// ```ignore
/// use bm3d_core::multiscale::{multiscale_bm3d_streak_removal, MultiscaleConfig};
/// use ndarray::Array2;
///
/// let sinogram = Array2::<f32>::zeros((256, 512));
/// let config = MultiscaleConfig::default();
/// let result = multiscale_bm3d_streak_removal(sinogram.view(), &config);
/// ```
pub fn multiscale_bm3d_streak_removal<F: Bm3dFloat>(
    sinogram: ArrayView2<F>,
    config: &MultiscaleConfig<F>,
) -> Result<Array2<F>, String> {
    // Validate configuration
    config.validate()?;

    let (rows, cols) = sinogram.dim();

    // Compute number of scales
    let num_scales = config.num_scales.unwrap_or_else(|| compute_num_scales(cols));

    // If K=0, just run single-scale BM3D
    if num_scales == 0 {
        return bm3d_ring_artifact_removal(sinogram, RingRemovalMode::Streak, &config.bm3d_config);
    }

    // Build horizontal pyramid (store originals at each scale)
    let mut pyramid_orig: Vec<Array2<F>> = Vec::with_capacity(num_scales + 1);
    pyramid_orig.push(sinogram.to_owned());

    for _ in 0..num_scales {
        let last = pyramid_orig.last().unwrap();
        let binned = bin_horizontal(last.view(), BINNING_FACTOR);
        pyramid_orig.push(binned);
    }

    // Prepare working pyramid (will be modified during processing)
    let mut pyramid_work: Vec<Array2<F>> = vec![Array2::zeros((0, 0)); num_scales + 1];
    pyramid_work[num_scales] = pyramid_orig[num_scales].clone();

    // Compute denoise sizes for PSD generation
    let denoise_sizes: Vec<usize> = pyramid_orig.iter().map(|arr| arr.ncols()).collect();

    // Generate PSD shapes for each scale
    // Note: Currently unused as we use the standard streak mode PSD construction.
    // Future enhancement: pass per-scale PSD shapes to BM3D for full fidelity.
    let _psd_shapes = generate_psd_shapes::<F>(&denoise_sizes);

    // Coarse-to-fine processing
    //
    // Key insight from paper (Mäkinen et al., Section 3.2):
    // Z*_k = Z_k - B_h^{-1}(Z_{k+1} - Ŷ_{k+1})
    //      = Z_k + B_h^{-1}(Ŷ_{k+1} - Z_{k+1})
    //      = Z_k + debinned_residual
    //
    // The residual (denoised - original) should be computed in ORIGINAL space
    // (where values are scaled by 2^k due to sum-convolution binning).
    // Normalization is ONLY applied immediately before/after BM3D call.
    let mut denoised: Option<Array2<F>> = None;

    for scale in (0..=num_scales).rev() {
        // Get working image at this scale (in original scaled space)
        let img = pyramid_work[scale].clone();

        // Compute normalization factor: 2^scale
        // Sum-convolution binning scales values by 2^k at scale k
        let norm_factor = F::from_f64_c(2.0f64.powi(scale as i32));

        // Check if image is large enough for BM3D
        if img.nrows() < config.bm3d_config.patch_size
            || img.ncols() < config.bm3d_config.patch_size
        {
            // Image too small for BM3D at this scale, skip
            // Pass through the image as-is (no denoising, but propagate to finer scale)
            if scale > 0 {
                // For skip path, residual captures accumulated changes from coarser scales
                let residual = &img - &pyramid_orig[scale];

                // Scale correction: residual is in 2^scale space, target is in 2^(scale-1) space
                let scale_correction = F::from_f64_c(BINNING_FACTOR as f64);
                let residual_corrected = residual.mapv(|x| x / scale_correction);

                // Debin residual to finer scale
                let target_width = pyramid_orig[scale - 1].ncols();
                let debinned_residual = debin_horizontal(
                    residual_corrected.view(),
                    target_width,
                    BINNING_FACTOR,
                    config.debin_iterations,
                );

                // Ensure dimensions match
                let target_rows = pyramid_orig[scale - 1].nrows();
                let mut adjusted_residual = Array2::zeros((target_rows, target_width));
                let copy_rows = debinned_residual.nrows().min(target_rows);
                let copy_cols = debinned_residual.ncols().min(target_width);
                adjusted_residual
                    .slice_mut(s![..copy_rows, ..copy_cols])
                    .assign(&debinned_residual.slice(s![..copy_rows, ..copy_cols]));

                // Add residual to finer scale's original (both now in same space)
                pyramid_work[scale - 1] = &pyramid_orig[scale - 1] + &adjusted_residual;
            }
            denoised = Some(img);
            continue;
        }

        // Create per-scale BM3D config with appropriate threshold
        // Apply filter_strength multiplier for multi-scale processing
        // (num_scales=0 is handled by early return at line 481)
        let mut scale_config = config.bm3d_config.clone();
        scale_config.threshold = config.threshold * config.filter_strength;

        // === NORMALIZE only for BM3D call ===
        let img_normalized = img.mapv(|x| x / norm_factor);

        // === ADAPTIVE SIGMA ESTIMATION (Mäkinen et al., Section 3.3.2) ===
        // Estimate noise sigma on the normalized image using MAD-based robust estimation.
        // This allows BM3D to adapt its filtering strength to the actual noise level
        // at each scale, preventing over-filtering (low PSNR) or under-filtering.
        let estimated_sigma = estimate_sigma_mad(img_normalized.view());

        // Use estimated sigma for BM3D, with a minimum to avoid numerical issues
        // The estimation captures the actual noise level after all the residual propagation
        let min_sigma = F::from_f64_c(0.001);
        if estimated_sigma > min_sigma {
            scale_config.sigma_random = estimated_sigma;
        }

        // Denoise normalized image
        let den_normalized =
            bm3d_ring_artifact_removal(img_normalized.view(), RingRemovalMode::Streak, &scale_config)?;

        // === DENORMALIZE immediately after BM3D ===
        let den = den_normalized.mapv(|x| x * norm_factor);

        if scale > 0 {
            // Compute residual: what BM3D actually changed at this scale
            // Per paper: residual = Ŷ_k - Z*_k (denoised minus working, NOT original)
            // The working image `img` may already have residuals from coarser scales
            let residual = &den - &img;

            // Scale correction: residual is in 2^scale space, target is in 2^(scale-1) space
            // Since binning uses sum (not average), we need to divide by factor when debinning
            let scale_correction = F::from_f64_c(BINNING_FACTOR as f64);
            let residual_corrected = residual.mapv(|x| x / scale_correction);

            // Debin residual to finer scale
            let target_width = pyramid_orig[scale - 1].ncols();
            let debinned_residual = debin_horizontal(
                residual_corrected.view(),
                target_width,
                BINNING_FACTOR,
                config.debin_iterations,
            );

            // Ensure dimensions match
            let target_rows = pyramid_orig[scale - 1].nrows();
            let mut adjusted_residual = Array2::zeros((target_rows, target_width));
            let copy_rows = debinned_residual.nrows().min(target_rows);
            let copy_cols = debinned_residual.ncols().min(target_width);
            adjusted_residual
                .slice_mut(s![..copy_rows, ..copy_cols])
                .assign(&debinned_residual.slice(s![..copy_rows, ..copy_cols]));

            // Add residual to finer scale's original (both now in same space)
            // This implements: Z*_{k-1} = Z_{k-1} + B_h^{-1}(Ŷ_k - Z_k)
            pyramid_work[scale - 1] = &pyramid_orig[scale - 1] + &adjusted_residual;
        }

        // Store result (already in original space)
        denoised = Some(den);
    }

    // The final denoised result is at the finest scale
    // But we need to account for the residual propagation
    // The actual output should match the original dimensions
    let final_result = denoised.ok_or("No scales processed")?;

    // Ensure output matches input dimensions
    if final_result.dim() != (rows, cols) {
        // This shouldn't happen, but handle gracefully
        let mut output = Array2::zeros((rows, cols));
        let copy_rows = final_result.nrows().min(rows);
        let copy_cols = final_result.ncols().min(cols);
        output
            .slice_mut(s![..copy_rows, ..copy_cols])
            .assign(&final_result.slice(s![..copy_rows, ..copy_cols]));
        Ok(output)
    } else {
        Ok(final_result)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // Helper: Simple LCG for deterministic test data
    struct SimpleLcg {
        state: u64,
    }

    impl SimpleLcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
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

    // ==================== Scale Computation Tests ====================

    #[test]
    fn test_compute_num_scales_small() {
        // Width <= 40 should give K=0
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
        // Width 640: floor(log2(640/40)) = floor(log2(16)) = 4
        assert_eq!(compute_num_scales(640), 4);
        // Width 1280: floor(log2(1280/40)) = floor(log2(32)) = 5
        assert_eq!(compute_num_scales(1280), 5);
    }

    // ==================== Binning Tests ====================

    #[test]
    fn test_binning_reduces_width() {
        let arr = random_matrix(32, 100, 12345);
        let binned = bin_horizontal(arr.view(), 2);

        // Width should be approximately halved
        assert!(binned.ncols() <= 51); // 100/2 + 1 for padding edge cases
        assert!(binned.ncols() >= 49);
        assert_eq!(binned.nrows(), 32); // Rows unchanged
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
        // Binning uses sum-convolution, so total sum should be roughly preserved
        // (modulo boundary effects)
        let arr = random_matrix(16, 64, 22222);
        let sum_orig: f32 = arr.iter().sum();

        let binned = bin_horizontal(arr.view(), 2);
        let sum_binned: f32 = binned.iter().sum();

        // Sum should be in same order of magnitude
        // (not exactly equal due to boundary handling)
        assert!(
            (sum_binned - sum_orig).abs() < sum_orig * 0.5,
            "Sum changed too much: {} vs {}",
            sum_binned,
            sum_orig
        );
    }

    // ==================== Debinning Tests ====================

    #[test]
    fn test_debin_restores_width() {
        let original = random_matrix(32, 100, 33333);
        let binned = bin_horizontal(original.view(), 2);

        let debinned = debin_horizontal(binned.view(), 100, 2, 30);

        assert_eq!(debinned.dim(), (32, 100));
    }

    #[test]
    fn test_bin_debin_approximate_recovery() {
        // Binning followed by debinning should approximately recover original
        let original = random_matrix(16, 64, 44444);
        let binned = bin_horizontal(original.view(), 2);
        let debinned = debin_horizontal(binned.view(), 64, 2, 30);

        // Compute mean absolute error
        let mae: f32 = original
            .iter()
            .zip(debinned.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / (original.len() as f32);

        // Should be reasonably close (within 50% of mean value)
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

    // ==================== PSD Shape Tests ====================

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
        // Coarsest scale should have flat (constant) PSD
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

    // ==================== Config Tests ====================

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
        let mut config: MultiscaleConfig<f32> = MultiscaleConfig::default();
        config.filter_strength = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_debin_iterations() {
        let mut config: MultiscaleConfig<f32> = MultiscaleConfig::default();
        config.debin_iterations = 0;
        assert!(config.validate().is_err());
    }

    // ==================== Smoke Tests ====================

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
    fn test_k0_equals_single_scale() {
        // For small images (K=0), multiscale should equal single-scale
        let image = random_matrix(32, 32, 88888);
        let config = MultiscaleConfig::default();

        let multiscale_result = multiscale_bm3d_streak_removal(image.view(), &config).unwrap();

        let single_result =
            bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Streak, &config.bm3d_config)
                .unwrap();

        // Results should be identical for K=0
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
        // For larger images, multiscale should differ from single-scale
        let image = random_matrix(64, 256, 99999);

        let mut single_config: Bm3dConfig<f32> = Bm3dConfig::default();
        single_config.threshold = 3.5; // Match multiscale default

        let single_result =
            bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Streak, &single_config)
                .unwrap();

        let multi_config = MultiscaleConfig::default();
        let multi_result = multiscale_bm3d_streak_removal(image.view(), &multi_config).unwrap();

        // Results should differ
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

    // ==================== Wide Streak Test ====================

    #[test]
    fn test_multiscale_handles_wide_streak() {
        // Create image with wide vertical streak
        let mut image = Array2::from_elem((64, 256), 0.5f32);

        // Add a wide streak (10 pixels wide)
        for r in 0..64 {
            for c in 120..130 {
                image[[r, c]] = 0.9;
            }
        }

        let config = MultiscaleConfig::default();
        let result = multiscale_bm3d_streak_removal(image.view(), &config);

        assert!(result.is_ok());
        let output = result.unwrap();

        // Compute column variance - should be reduced
        let col_means: Vec<f32> = (0..256)
            .map(|c| {
                let sum: f32 = (0..64).map(|r| output[[r, c]]).sum();
                sum / 64.0
            })
            .collect();

        let overall_mean: f32 = col_means.iter().sum::<f32>() / 256.0;
        let col_variance: f32 =
            col_means.iter().map(|m| (m - overall_mean).powi(2)).sum::<f32>() / 256.0;

        // Original column variance
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

        // Multiscale should reduce column variance from wide streak
        assert!(
            col_variance < orig_col_variance,
            "Multiscale should reduce wide streak variance: {} >= {}",
            col_variance,
            orig_col_variance
        );
    }

    // ==================== f64 Tests ====================

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

    // ==================== Override Scale Count ====================

    #[test]
    fn test_override_num_scales() {
        let image = random_matrix(64, 256, 10101);

        // Force K=1 even though auto would compute higher
        let mut config: MultiscaleConfig<f32> = MultiscaleConfig::default();
        config.num_scales = Some(1);

        let result = multiscale_bm3d_streak_removal(image.view(), &config);

        assert!(result.is_ok());
    }

    #[test]
    fn test_force_k0() {
        // Force K=0 on large image
        let image = random_matrix(64, 256, 20202);

        let mut config: MultiscaleConfig<f32> = MultiscaleConfig::default();
        config.num_scales = Some(0);

        let multi_result = multiscale_bm3d_streak_removal(image.view(), &config).unwrap();

        let single_result = bm3d_ring_artifact_removal(
            image.view(),
            RingRemovalMode::Streak,
            &config.bm3d_config,
        )
        .unwrap();

        // Should be identical when forcing K=0
        for (a, b) in multi_result.iter().zip(single_result.iter()) {
            assert!(approx_eq(*a, *b, 1e-5), "Forced K=0 differs from single");
        }
    }
}
