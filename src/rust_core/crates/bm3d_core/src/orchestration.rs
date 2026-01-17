//! Unified BM3D Ring Artifact Removal Pipeline
//!
//! This module provides a complete BM3D denoising pipeline for 2D sinograms,
//! eliminating the need for Python orchestration. It combines:
//! - Normalization / denormalization
//! - Sigma map computation from streak profile
//! - Optional streak pre-subtraction (streak mode)
//! - Two-pass BM3D: Hard Threshold → Wiener
//!
//! ## Modes
//!
//! - **Generic**: Standard BM3D assuming white noise
//! - **Streak**: Streak pre-subtraction + anisotropic PSD for ring artifact removal

use ndarray::{Array1, Array2, ArrayView2};

use crate::float_trait::Bm3dFloat;
use crate::noise_estimation::estimate_noise_sigma;
use crate::pipeline::{run_bm3d_step, Bm3dMode};
use crate::streak::{estimate_streak_profile_impl, gaussian_blur_1d};

// =============================================================================
// Constants
// =============================================================================

/// Default random noise standard deviation (0.0 = auto-estimate)
const DEFAULT_SIGMA_RANDOM: f64 = 0.0;

/// Default patch size for block matching
const DEFAULT_PATCH_SIZE: usize = 8;

/// Default step size (stride) between patches
const DEFAULT_STEP_SIZE: usize = 4;

/// Default search window size for block matching
const DEFAULT_SEARCH_WINDOW: usize = 24;

/// Default maximum number of similar patches per group
const DEFAULT_MAX_MATCHES: usize = 16;

/// Default hard thresholding coefficient
const DEFAULT_THRESHOLD: f64 = 2.7;

/// Default sigma for streak estimation smoothing
const DEFAULT_STREAK_SIGMA_SMOOTH: f64 = 3.0;

/// Default number of streak estimation iterations
const DEFAULT_STREAK_ITERATIONS: usize = 2;

/// Default sigma for sigma map profile smoothing
const DEFAULT_SIGMA_MAP_SMOOTHING: f64 = 20.0;

/// Default streak sigma scale factor
const DEFAULT_STREAK_SIGMA_SCALE: f64 = 1.1;

/// Default PSD Gaussian width for streak mode
const DEFAULT_PSD_WIDTH: f64 = 0.6;

/// Default filter strength multiplier
const DEFAULT_FILTER_STRENGTH: f64 = 1.0;

/// Fixed sigma for streak profile estimation when computing sigma map
const SIGMA_MAP_STREAK_SIGMA: f64 = 5.0;

/// Fixed iterations for streak profile estimation when computing sigma map
const SIGMA_MAP_STREAK_ITERATIONS: usize = 1;

/// Default FFT alpha (trust factor). 0.0 = disabled, 1.0 = standard boost.
const DEFAULT_FFT_ALPHA: f64 = 1.0;

/// Default Gaussian notch width for vertical energy detection
const DEFAULT_NOTCH_WIDTH: f64 = 2.0;

/// Small epsilon to avoid division by zero during normalization
const NORMALIZATION_EPSILON: f64 = 1e-10;

// =============================================================================
// Types
// =============================================================================

/// Processing mode for BM3D ring artifact removal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RingRemovalMode {
    /// Standard BM3D assuming white noise.
    /// No streak pre-subtraction, uses scalar PSD.
    Generic,
    /// Streak pre-subtraction + anisotropic PSD.
    /// Designed for ring artifact removal in sinograms.
    Streak,
    /// Multi-scale BM3D with streak pre-subtraction.
    /// Handles wide streaks via pyramid processing.
    #[default]
    MultiscaleStreak,
    /// Fourier-SVD algorithm.
    /// Uses FFT-guided energy detection with rank-1 SVD for streak removal.
    FourierSvd,
}

/// Configuration for BM3D ring artifact removal.
///
/// All parameters have sensible defaults matching the original Python implementation.
/// Use `Default::default()` for standard settings.
#[derive(Debug, Clone)]
pub struct Bm3dConfig<F: Bm3dFloat> {
    /// Random noise standard deviation. Default: 0.1
    pub sigma_random: F,
    /// Block matching patch size. Default: 8
    pub patch_size: usize,
    /// Stride between patches. Default: 4
    pub step_size: usize,
    /// Search window size for block matching. Default: 24
    pub search_window: usize,
    /// Maximum similar patches per group. Default: 16
    pub max_matches: usize,
    /// Hard thresholding coefficient. Default: 2.7
    pub threshold: F,
    /// Sigma for streak estimation smoothing. Default: 3.0
    pub streak_sigma_smooth: F,
    /// Number of streak estimation iterations. Default: 2
    pub streak_iterations: usize,
    /// Sigma for sigma map profile smoothing. Default: 20.0
    pub sigma_map_smoothing: F,
    /// Sigma map scaling factor. Default: 1.1
    pub streak_sigma_scale: F,
    /// PSD Gaussian width for streak mode. Default: 0.6
    pub psd_width: F,
    /// Filter strength multiplier (affects BM3D thresholding). Default: 1.0
    pub filter_strength: F,
    /// FFT-Guided SVD: Trust factor (Alpha). Default: 1.0
    pub fft_alpha: F,
    /// FFT-Guided SVD: Gaussian notch width. Default: 2.0
    pub notch_width: F,
}

impl<F: Bm3dFloat> Default for Bm3dConfig<F> {
    fn default() -> Self {
        Self {
            sigma_random: F::from_f64_c(DEFAULT_SIGMA_RANDOM),
            patch_size: DEFAULT_PATCH_SIZE,
            step_size: DEFAULT_STEP_SIZE,
            search_window: DEFAULT_SEARCH_WINDOW,
            max_matches: DEFAULT_MAX_MATCHES,
            threshold: F::from_f64_c(DEFAULT_THRESHOLD),
            streak_sigma_smooth: F::from_f64_c(DEFAULT_STREAK_SIGMA_SMOOTH),
            streak_iterations: DEFAULT_STREAK_ITERATIONS,
            sigma_map_smoothing: F::from_f64_c(DEFAULT_SIGMA_MAP_SMOOTHING),
            streak_sigma_scale: F::from_f64_c(DEFAULT_STREAK_SIGMA_SCALE),
            psd_width: F::from_f64_c(DEFAULT_PSD_WIDTH),
            filter_strength: F::from_f64_c(DEFAULT_FILTER_STRENGTH),
            fft_alpha: F::from_f64_c(DEFAULT_FFT_ALPHA),
            notch_width: F::from_f64_c(DEFAULT_NOTCH_WIDTH),
        }
    }
}

impl<F: Bm3dFloat> Bm3dConfig<F> {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate the configuration parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.patch_size == 0 {
            return Err("patch_size must be > 0".to_string());
        }
        if self.step_size == 0 {
            return Err("step_size must be > 0".to_string());
        }
        if self.search_window == 0 {
            return Err("search_window must be > 0".to_string());
        }
        if self.max_matches == 0 {
            return Err("max_matches must be > 0".to_string());
        }
        if self.sigma_random < F::zero() {
            return Err("sigma_random must be >= 0".to_string());
        }
        if self.threshold < F::zero() {
            return Err("threshold must be >= 0".to_string());
        }
        if self.filter_strength <= F::zero() {
            return Err("filter_strength must be > 0".to_string());
        }
        if self.fft_alpha < F::zero() {
            return Err("fft_alpha must be >= 0".to_string());
        }
        if self.notch_width <= F::zero() {
            return Err("notch_width must be > 0".to_string());
        }
        Ok(())
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute the spatially adaptive sigma map from the input image.
///
/// The sigma map captures local noise structure by:
/// 1. Estimating streak profile (sigma=5.0, iterations=1)
/// 2. Smoothing the profile (sigma=20.0)
/// 3. Computing streak signal as difference
/// 4. Scaling absolute streak signal
/// 5. Tiling 1D result to full image height
fn compute_sigma_map<F: Bm3dFloat>(
    normalized_image: ArrayView2<F>,
    sigma_map_smoothing: F,
    streak_sigma_scale: F,
) -> Array2<F> {
    let (rows, cols) = normalized_image.dim();

    // 1. Estimate streak profile with fixed parameters
    let streak_profile = estimate_streak_profile_impl(
        normalized_image,
        F::from_f64_c(SIGMA_MAP_STREAK_SIGMA),
        SIGMA_MAP_STREAK_ITERATIONS,
    );

    // 2. Smooth the profile
    let profile_smooth = gaussian_blur_1d(streak_profile.view(), sigma_map_smoothing);

    // 3. Compute streak signal (high-frequency component)
    let streak_signal: Array1<F> = &streak_profile - &profile_smooth;

    // 4. Take absolute value and scale
    let sigma_1d: Array1<F> = streak_signal.mapv(|x| x.abs() * streak_sigma_scale);

    // 5. Tile to full image height
    let mut sigma_map = Array2::zeros((rows, cols));
    for r in 0..rows {
        sigma_map.row_mut(r).assign(&sigma_1d);
    }

    sigma_map
}

/// Construct anisotropic PSD for streak mode.
///
/// Creates a 2D array with Gaussian profile along Y-axis (rows),
/// replicated across X-axis (columns). This models vertical streak noise.
fn construct_psd<F: Bm3dFloat>(patch_size: usize, psd_width: F) -> Array2<F> {
    let mut sigma_psd = Array2::zeros((patch_size, patch_size));

    // Gaussian profile along Y-axis
    let neg_half = F::from_f64_c(-0.5);
    for y in 0..patch_size {
        let y_f = F::usize_as(y);
        let normalized = y_f / psd_width;
        let value = (neg_half * normalized * normalized).exp();

        // Replicate across X-axis
        for x in 0..patch_size {
            sigma_psd[[y, x]] = value;
        }
    }

    sigma_psd
}

/// Subtract streak profile from the image (in-place operation on owned array).
fn subtract_streak_profile<F: Bm3dFloat>(
    image: &mut Array2<F>,
    streak_sigma_smooth: F,
    streak_iterations: usize,
) {
    let (rows, _cols) = image.dim();

    // Estimate streak profile
    let profile =
        estimate_streak_profile_impl(image.view(), streak_sigma_smooth, streak_iterations);

    // Subtract from each row (broadcast)
    for r in 0..rows {
        let mut row = image.row_mut(r);
        for (c, val) in row.iter_mut().enumerate() {
            *val -= profile[c];
        }
    }
}

// =============================================================================
// Main Entry Point
// =============================================================================

/// Unified BM3D ring artifact removal with pre-computed plans (Internal/Advanced).
///
/// This is the core implementation that takes pre-computed FFT plans.
/// See `bm3d_ring_artifact_removal` for the public API.
pub fn bm3d_ring_artifact_removal_with_plans<F: Bm3dFloat>(
    sinogram: ArrayView2<F>,
    mode: RingRemovalMode,
    config: &Bm3dConfig<F>,
    plans: &crate::pipeline::Bm3dPlans<F>,
) -> Result<Array2<F>, String> {
    // Validate configuration
    config.validate()?;

    let (rows, cols) = sinogram.dim();

    // Check image is large enough for patch size
    if rows < config.patch_size || cols < config.patch_size {
        return Err(format!(
            "Image size ({}, {}) is smaller than patch_size {}",
            rows, cols, config.patch_size
        ));
    }

    // Step 1: Compute global min/max for normalization
    let d_min = sinogram
        .iter()
        .copied()
        .fold(F::infinity(), |a, b| if b < a { b } else { a });
    let d_max = sinogram
        .iter()
        .copied()
        .fold(F::neg_infinity(), |a, b| if b > a { b } else { a });

    let range = d_max - d_min;
    let eps = F::from_f64_c(NORMALIZATION_EPSILON);

    // Step 2: Normalize to [0, 1]
    let mut z_norm = if range > eps {
        sinogram.mapv(|x| (x - d_min) / range)
    } else {
        // Constant image - just use zeros
        Array2::zeros((rows, cols))
    };

    // Step 3: Compute sigma map
    let sigma_map = compute_sigma_map(
        z_norm.view(),
        config.sigma_map_smoothing,
        config.streak_sigma_scale,
    );

    // Step 4: Prepare PSD based on mode
    let sigma_psd = match mode {
        RingRemovalMode::Generic => {
            // Scalar PSD (no colored noise model)
            Array2::zeros((1, 1))
        }
        RingRemovalMode::Streak | RingRemovalMode::MultiscaleStreak => {
            // Anisotropic PSD for streak modes
            construct_psd(config.patch_size, config.psd_width)
        }
        RingRemovalMode::FourierSvd => {
            // Fourier-SVD: No PSD needed (uses separate algorithm)
            Array2::zeros((1, 1))
        }
    };

    // Step 5: Streak pre-subtraction (streak modes) or Fourier-SVD
    if mode == RingRemovalMode::Streak || mode == RingRemovalMode::MultiscaleStreak {
        subtract_streak_profile(
            &mut z_norm,
            config.streak_sigma_smooth,
            config.streak_iterations,
        );
    } else if mode == RingRemovalMode::FourierSvd {
        // Fourier-SVD Streak Removal
        z_norm = crate::fourier_svd::fourier_svd_removal(
            z_norm.view(),
            config.fft_alpha,
            config.notch_width,
        );
    }

    // Step 5b: Auto-estimate sigma if not provided
    // If sigma_random is effectively 0, estimate from data.
    let sigma_random = if config.sigma_random <= F::from_f64_c(1e-6) {
        // Use z_norm for estimation (it has streaks removed if in Streak/FourierSvd mode)
        estimate_noise_sigma(z_norm.view())
    } else {
        config.sigma_random
    };

    // Step 6: BM3D Pass 1 - Hard Thresholding
    let yhat_ht = run_bm3d_step(
        z_norm.view(),
        z_norm.view(), // pilot = noisy for first pass
        Bm3dMode::HardThreshold,
        sigma_psd.view(),
        sigma_map.view(),
        sigma_random,
        config.threshold,
        config.patch_size,
        config.step_size,
        config.search_window,
        config.max_matches,
        plans,
    )?;

    // Step 7: BM3D Pass 2 - Wiener Filtering
    let yhat_final = run_bm3d_step(
        z_norm.view(),
        yhat_ht.view(), // pilot = HT result
        Bm3dMode::Wiener,
        sigma_psd.view(),
        sigma_map.view(),
        sigma_random,
        F::zero(), // threshold not used for Wiener
        config.patch_size,
        config.step_size,
        config.search_window,
        config.max_matches,
        plans,
    )?;

    // Step 8: Denormalize to original range
    let output = if range > eps {
        yhat_final.mapv(|x| x * range + d_min)
    } else {
        // Constant image - restore original mean/min
        Array2::from_elem(yhat_final.raw_dim(), d_min)
    };

    Ok(output)
}

/// Unified BM3D ring artifact removal for 2D sinograms.
///
/// This function performs the complete BM3D denoising pipeline:
/// 1. Normalizes input to [0, 1] range
/// 2. Computes spatially adaptive sigma map
/// 3. (Streak mode) Subtracts estimated streak profile
/// 4. Runs two-pass BM3D: Hard Threshold → Wiener
/// 5. Denormalizes output to original range
///
/// # Arguments
///
/// * `sinogram` - Input 2D sinogram (H × W)
/// * `mode` - Processing mode (Generic or Streak)
/// * `config` - Configuration parameters
///
/// # Returns
///
/// Denoised sinogram with same shape as input, or error if parameters invalid.
///
/// # Example
///
/// ```
/// use bm3d_core::{bm3d_ring_artifact_removal, RingRemovalMode, Bm3dConfig};
/// use ndarray::Array2;
///
/// let sinogram = Array2::<f32>::zeros((64, 64));
/// let config = Bm3dConfig::default();
/// let result = bm3d_ring_artifact_removal(sinogram.view(), RingRemovalMode::Generic, &config);
/// assert!(result.is_ok());
/// ```
pub fn bm3d_ring_artifact_removal<F: Bm3dFloat>(
    sinogram: ArrayView2<F>,
    mode: RingRemovalMode,
    config: &Bm3dConfig<F>,
) -> Result<Array2<F>, String> {
    // Create plans locally
    let plans = crate::pipeline::Bm3dPlans::new(config.patch_size, config.max_matches);
    bm3d_ring_artifact_removal_with_plans(sinogram, mode, config, &plans)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
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

    // ==================== Config Tests ====================

    #[test]
    fn test_default_config_matches_spec() {
        let config: Bm3dConfig<f32> = Bm3dConfig::default();

        assert!(approx_eq(config.sigma_random, 0.0, 1e-6));
        assert_eq!(config.patch_size, 8);
        assert_eq!(config.step_size, 4);
        assert_eq!(config.search_window, 24);
        assert_eq!(config.max_matches, 16);
        assert!(approx_eq(config.threshold, 2.7, 1e-6));
        assert!(approx_eq(config.streak_sigma_smooth, 3.0, 1e-6));
        assert_eq!(config.streak_iterations, 2);
        assert!(approx_eq(config.sigma_map_smoothing, 20.0, 1e-6));
        assert!(approx_eq(config.streak_sigma_scale, 1.1, 1e-6));
        assert!(approx_eq(config.psd_width, 0.6, 1e-6));
        assert!(approx_eq(config.filter_strength, 1.0, 1e-6));
    }

    #[test]
    fn test_config_validation_valid() {
        let config: Bm3dConfig<f32> = Bm3dConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_invalid_patch_size() {
        let mut config: Bm3dConfig<f32> = Bm3dConfig::default();
        config.patch_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_step_size() {
        let mut config: Bm3dConfig<f32> = Bm3dConfig::default();
        config.step_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_negative_sigma() {
        let mut config: Bm3dConfig<f32> = Bm3dConfig::default();
        config.sigma_random = -0.1;
        assert!(config.validate().is_err());
    }

    // ==================== Sigma Map Tests ====================

    #[test]
    fn test_sigma_map_shape() {
        let image = random_matrix(64, 128, 12345);
        let normalized = image.mapv(|x| x); // Already in [0,1]

        let sigma_map = compute_sigma_map(normalized.view(), 20.0, 1.1);

        assert_eq!(sigma_map.dim(), (64, 128));
    }

    #[test]
    fn test_sigma_map_rows_identical() {
        // Sigma map should have identical rows (tiled from 1D profile)
        let image = random_matrix(32, 64, 54321);
        let sigma_map = compute_sigma_map(image.view(), 20.0, 1.1);

        let first_row: Vec<f32> = sigma_map.row(0).iter().copied().collect();
        for r in 1..32 {
            let row: Vec<f32> = sigma_map.row(r).iter().copied().collect();
            for (a, b) in first_row.iter().zip(row.iter()) {
                assert!(approx_eq(*a, *b, 1e-6), "Row {} differs from row 0", r);
            }
        }
    }

    #[test]
    fn test_sigma_map_non_negative() {
        let image = random_matrix(32, 64, 11111);
        let sigma_map = compute_sigma_map(image.view(), 20.0, 1.1);

        for &val in sigma_map.iter() {
            assert!(val >= 0.0, "Sigma map should be non-negative");
        }
    }

    // ==================== PSD Tests ====================

    #[test]
    fn test_psd_shape() {
        let psd = construct_psd::<f32>(8, 0.6);
        assert_eq!(psd.dim(), (8, 8));
    }

    #[test]
    fn test_psd_columns_identical() {
        // PSD should have identical columns (replicated Gaussian profile)
        let psd = construct_psd::<f32>(8, 0.6);

        let first_col: Vec<f32> = (0..8).map(|r| psd[[r, 0]]).collect();
        for c in 1..8 {
            let col: Vec<f32> = (0..8).map(|r| psd[[r, c]]).collect();
            for (a, b) in first_col.iter().zip(col.iter()) {
                assert!(
                    approx_eq(*a, *b, 1e-6),
                    "Column {} differs from column 0",
                    c
                );
            }
        }
    }

    #[test]
    fn test_psd_gaussian_profile() {
        // First element (y=0) should be 1.0 (exp(0))
        let psd = construct_psd::<f32>(8, 0.6);
        assert!(approx_eq(psd[[0, 0]], 1.0, 1e-6));

        // Values should decrease as y increases
        for y in 1..8 {
            assert!(psd[[y, 0]] < psd[[y - 1, 0]], "PSD should decrease with y");
        }
    }

    #[test]
    fn test_psd_all_positive() {
        let psd = construct_psd::<f32>(8, 0.6);
        for &val in psd.iter() {
            assert!(val > 0.0, "PSD values should be positive");
        }
    }

    // ==================== Smoke Tests ====================

    #[test]
    fn test_generic_mode_smoke() {
        let image = random_matrix(32, 32, 12345);
        let config = Bm3dConfig::default();

        let result = bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Generic, &config);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.dim(), image.dim());
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_streak_mode_smoke() {
        let image = random_matrix(32, 32, 54321);
        let config = Bm3dConfig::default();

        let result = bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Streak, &config);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.dim(), image.dim());
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    // ==================== Shape Preservation ====================

    #[test]
    fn test_output_shape_matches_input() {
        let config = Bm3dConfig::default();

        for (rows, cols) in [(32, 32), (48, 64), (64, 48)] {
            let image = random_matrix(rows, cols, (rows * 100 + cols) as u64);
            let result =
                bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Generic, &config);

            assert!(result.is_ok());
            assert_eq!(
                result.unwrap().dim(),
                (rows, cols),
                "Shape mismatch for {}x{}",
                rows,
                cols
            );
        }
    }

    // ==================== Normalization Tests ====================

    #[test]
    fn test_handles_non_normalized_input() {
        // Input outside [0,1] range
        let image = Array2::from_shape_fn((32, 32), |(r, c)| 100.0 + (r * 32 + c) as f32 * 10.0);
        let config = Bm3dConfig::default();

        let result = bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Generic, &config);

        assert!(result.is_ok());
        let output = result.unwrap();

        // Output should be in similar range as input
        let out_min = output.iter().copied().fold(f32::INFINITY, f32::min);
        let out_max = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let in_min = image.iter().copied().fold(f32::INFINITY, f32::min);
        let in_max = image.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Output range should be similar to input range (within reason)
        assert!(
            out_min >= in_min - (in_max - in_min) * 0.5,
            "Output min {} too low compared to input min {}",
            out_min,
            in_min
        );
        assert!(
            out_max <= in_max + (in_max - in_min) * 0.5,
            "Output max {} too high compared to input max {}",
            out_max,
            in_max
        );
    }

    #[test]
    fn test_constant_image_unchanged() {
        // Constant image should remain approximately constant
        let constant_val = 42.5f32;
        let image = Array2::from_elem((32, 32), constant_val);
        let config = Bm3dConfig::default();

        let result = bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Generic, &config);

        assert!(result.is_ok());
        let output = result.unwrap();

        // All values should be the constant
        for &val in output.iter() {
            assert!(
                approx_eq(val, constant_val, 1e-5),
                "Constant image should remain constant, got {}",
                val
            );
        }
    }

    // ==================== Mode Behavior Tests ====================

    #[test]
    fn test_generic_and_streak_differ() {
        // Create image with vertical streak
        let mut image = Array2::from_elem((32, 64), 0.5f32);
        for r in 0..32 {
            image[[r, 20]] = 0.9; // Bright vertical stripe
        }

        let config = Bm3dConfig::default();

        let result_generic =
            bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Generic, &config).unwrap();
        let result_streak =
            bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Streak, &config).unwrap();

        // Outputs should differ
        let diff: f32 = result_generic
            .iter()
            .zip(result_streak.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(
            diff > 0.01,
            "Generic and streak modes should produce different results"
        );
    }

    #[test]
    fn test_streak_mode_reduces_vertical_artifacts() {
        // Create image with strong vertical streak
        let mut image = Array2::from_elem((64, 64), 0.5f32);
        for r in 0..64 {
            image[[r, 32]] = 1.0; // Bright vertical stripe in middle
        }

        let config = Bm3dConfig::default();

        let result =
            bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Streak, &config).unwrap();

        // The streak should be reduced
        // Compute column variance - streak mode should reduce it
        let col_means: Vec<f32> = (0..64)
            .map(|c| {
                let sum: f32 = (0..64).map(|r| result[[r, c]]).sum();
                sum / 64.0
            })
            .collect();

        let overall_mean: f32 = col_means.iter().sum::<f32>() / 64.0;
        let col_variance: f32 = col_means
            .iter()
            .map(|m| (m - overall_mean).powi(2))
            .sum::<f32>()
            / 64.0;

        // Original has high column variance due to streak
        let orig_col_means: Vec<f32> = (0..64)
            .map(|c| {
                let sum: f32 = (0..64).map(|r| image[[r, c]]).sum();
                sum / 64.0
            })
            .collect();
        let orig_overall_mean: f32 = orig_col_means.iter().sum::<f32>() / 64.0;
        let orig_col_variance: f32 = orig_col_means
            .iter()
            .map(|m| (m - orig_overall_mean).powi(2))
            .sum::<f32>()
            / 64.0;

        // Streak mode should reduce column variance
        assert!(
            col_variance < orig_col_variance,
            "Streak mode should reduce column variance: {} >= {}",
            col_variance,
            orig_col_variance
        );
    }

    // ==================== Error Handling Tests ====================

    #[test]
    fn test_image_too_small() {
        let image = random_matrix(4, 4, 99999);
        let config = Bm3dConfig::default(); // patch_size = 8

        let result = bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Generic, &config);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("smaller than patch_size"));
    }

    #[test]
    fn test_invalid_config_rejected() {
        let image = random_matrix(32, 32, 88888);
        let mut config: Bm3dConfig<f32> = Bm3dConfig::default();
        config.patch_size = 0;

        let result = bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Generic, &config);

        assert!(result.is_err());
    }

    // ==================== f64 Tests ====================

    #[test]
    fn test_f64_generic_mode() {
        let image = Array2::from_shape_fn((32, 32), |(r, c)| (r * 32 + c) as f64 / 1024.0);
        let config: Bm3dConfig<f64> = Bm3dConfig::default();

        let result = bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Generic, &config);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.dim(), image.dim());
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_f64_streak_mode() {
        let image = Array2::from_shape_fn((32, 32), |(r, c)| (r * 32 + c) as f64 / 1024.0);
        let config: Bm3dConfig<f64> = Bm3dConfig::default();

        let result = bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Streak, &config);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.dim(), image.dim());
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    // ==================== Parameter Variation Tests ====================

    #[test]
    fn test_different_patch_sizes() {
        let image = random_matrix(48, 48, 11111);

        for patch_size in [4, 8] {
            let mut config: Bm3dConfig<f32> = Bm3dConfig::default();
            config.patch_size = patch_size;
            config.step_size = patch_size / 2;

            let result =
                bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Generic, &config);

            assert!(result.is_ok(), "Failed for patch_size={}", patch_size);
            assert!(result.unwrap().iter().all(|&x| x.is_finite()));
        }
    }

    #[test]
    fn test_different_sigma_random() {
        let image = random_matrix(32, 32, 22222);

        for sigma in [0.05f32, 0.1, 0.2] {
            let mut config: Bm3dConfig<f32> = Bm3dConfig::default();
            config.sigma_random = sigma;

            let result =
                bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Generic, &config);

            assert!(result.is_ok(), "Failed for sigma={}", sigma);
        }
    }
    #[test]
    fn test_auto_sigma_estimation() {
        // Create a noisy image
        let mut rng = SimpleLcg::new(999);
        let image = Array2::from_shape_fn((64, 64), |_| rng.next_f32());

        // Config with sigma=0.0 to trigger auto-estimation
        let mut config = Bm3dConfig::default();
        config.sigma_random = 0.0;

        let result = bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Generic, &config);

        assert!(result.is_ok(), "Auto-estimation should succeed");
        let output = result.unwrap();
        assert_eq!(output.dim(), image.dim());

        // Ensure processed image isn't identical to input (denoising occurred)
        let diff: f32 = output
            .iter()
            .zip(image.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.1,
            "Denoising should have occurred (diff: {})",
            diff
        );
    }
}
