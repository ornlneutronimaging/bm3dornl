use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::time::Instant;

use crate::float_trait::Bm3dFloat;
use crate::noise_estimation::estimate_noise_sigma;
use crate::pipeline::{
    Bm3dKernelConfig, Bm3dMode, check_image_size, check_step_size, run_bm3d_step,
};
use crate::streak::{estimate_streak_profile_impl, gaussian_blur_1d};

const DEFAULT_SIGMA_RANDOM: f64 = 0.1;

const DEFAULT_PATCH_SIZE: usize = 8;

const DEFAULT_STEP_SIZE: usize = 4;

const DEFAULT_SEARCH_WINDOW: usize = 24;

const DEFAULT_MAX_MATCHES: usize = 16;

const DEFAULT_THRESHOLD: f64 = 2.7;

const DEFAULT_STREAK_SIGMA_SMOOTH: f64 = 3.0;

const DEFAULT_STREAK_ITERATIONS: usize = 2;

const DEFAULT_SIGMA_MAP_SMOOTHING: f64 = 20.0;

const DEFAULT_STREAK_SIGMA_SCALE: f64 = 1.1;

const DEFAULT_PSD_WIDTH: f64 = 0.6;

const DEFAULT_FILTER_STRENGTH: f64 = 1.0;

const SIGMA_MAP_STREAK_SIGMA: f64 = 5.0;

const SIGMA_MAP_STREAK_ITERATIONS: usize = 1;

const DEFAULT_FFT_ALPHA: f64 = 1.0;

const DEFAULT_NOTCH_WIDTH: f64 = 2.0;

const NORMALIZATION_EPSILON: f64 = 1e-10;
const PROFILE_TIMING_ENV: &str = "BM3D_PROFILE_TIMING";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RingRemovalMode {
    /// Standard BM3D assuming white noise.
    /// No streak pre-subtraction, uses scalar PSD.
    Generic,
    /// Streak pre-subtraction + anisotropic PSD.
    /// Designed for ring artifact removal in sinograms.
    #[default]
    Streak,
    MultiscaleStreak,
    /// Uses FFT-guided energy detection with rank-1 SVD for streak removal.
    FourierSvd,
}

/// Configuration for BM3D ring artifact removal.
///
/// Use `Default::default()` for standard settings.
#[derive(Debug, Clone)]
pub struct Bm3dConfig<F: Bm3dFloat> {
    /// Random noise standard deviation. Default: 0.1
    pub sigma_random: F,
    /// Block matching patch size. Default: 8
    pub patch_size: usize,
    /// Stride between reference patches, at most `patch_size`. Default: 4
    pub step_size: usize,
    /// Search window size for block matching. Default: 24
    pub search_window: usize,
    /// Maximum similar patches per group. Default: 16
    pub max_matches: usize,
    pub search_step: Option<usize>,
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
    pub filter_strength: F,
    /// FFT-Guided SVD: Trust factor (Alpha). Default: 1.0
    pub fft_alpha: F,
    /// FFT-Guided SVD: Gaussian notch width. Default: 2.0
    pub notch_width: F,
    /// Optional per-call override for 8x8 Hadamard fast path.
    /// `Some(true/false)` takes precedence over env/global defaults.
    pub use_hadamard_fast_path: Option<bool>,
}

impl<F: Bm3dFloat> Default for Bm3dConfig<F> {
    fn default() -> Self {
        Self {
            sigma_random: F::from_f64_c(DEFAULT_SIGMA_RANDOM),
            patch_size: DEFAULT_PATCH_SIZE,
            step_size: DEFAULT_STEP_SIZE,
            search_window: DEFAULT_SEARCH_WINDOW,
            max_matches: DEFAULT_MAX_MATCHES,
            search_step: None,
            threshold: F::from_f64_c(DEFAULT_THRESHOLD),
            streak_sigma_smooth: F::from_f64_c(DEFAULT_STREAK_SIGMA_SMOOTH),
            streak_iterations: DEFAULT_STREAK_ITERATIONS,
            sigma_map_smoothing: F::from_f64_c(DEFAULT_SIGMA_MAP_SMOOTHING),
            streak_sigma_scale: F::from_f64_c(DEFAULT_STREAK_SIGMA_SCALE),
            psd_width: F::from_f64_c(DEFAULT_PSD_WIDTH),
            filter_strength: F::from_f64_c(DEFAULT_FILTER_STRENGTH),
            fft_alpha: F::from_f64_c(DEFAULT_FFT_ALPHA),
            notch_width: F::from_f64_c(DEFAULT_NOTCH_WIDTH),
            use_hadamard_fast_path: None,
        }
    }
}

impl<F: Bm3dFloat> Bm3dConfig<F> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.patch_size == 0 {
            return Err("patch_size must be > 0".to_string());
        }
        check_step_size(self.step_size, self.patch_size)?;
        if self.search_window == 0 {
            return Err("search_window must be > 0".to_string());
        }
        if self.max_matches == 0 {
            return Err("max_matches must be > 0".to_string());
        }
        if self.search_step == Some(0) {
            return Err("search_step must be > 0 (or None for the per-mode default)".to_string());
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
        if self.psd_width <= F::zero() {
            return Err("psd_width must be > 0".to_string());
        }
        Ok(())
    }

    fn resolved_search_step(&self, mode: RingRemovalMode) -> usize {
        self.search_step.unwrap_or(match mode {
            RingRemovalMode::Generic => 1,
            RingRemovalMode::Streak
            | RingRemovalMode::MultiscaleStreak
            | RingRemovalMode::FourierSvd => self.step_size,
        })
    }
}

fn profile_timing_enabled() -> bool {
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

fn compute_sigma_map<F: Bm3dFloat>(
    normalized_image: ArrayView2<F>,
    sigma_map_smoothing: F,
    streak_sigma_scale: F,
) -> Array2<F> {
    let (_rows, cols) = normalized_image.dim();

    let streak_profile = estimate_streak_profile_impl(
        normalized_image,
        F::from_f64_c(SIGMA_MAP_STREAK_SIGMA),
        SIGMA_MAP_STREAK_ITERATIONS,
    );

    let profile_smooth = gaussian_blur_1d(streak_profile.view(), sigma_map_smoothing);

    let streak_signal: Array1<F> = &streak_profile - &profile_smooth;

    let sigma_1d: Array1<F> = streak_signal.mapv(|x| x.abs() * streak_sigma_scale);

    let mut sigma_map = Array2::zeros((1, cols));
    sigma_map.row_mut(0).assign(&sigma_1d);

    sigma_map
}

fn construct_psd<F: Bm3dFloat>(
    patch_size: usize,
    psd_width: F,
    psd_x: Option<ArrayView1<F>>,
) -> Array2<F> {
    let mut sigma_psd = Array2::zeros((patch_size, patch_size));

    // Resample the horizontal profile onto the patch's frequency grid,
    // normalised to unit mean so it shapes the PSD without rescaling it.
    let x_gain: Vec<F> = match psd_x {
        None => vec![F::one(); patch_size],
        Some(profile) => {
            let n = profile.len();
            if n == 0 {
                vec![F::one(); patch_size]
            } else {
                let mut powers = Vec::with_capacity(patch_size);
                for x in 0..patch_size {
                    // Circular frequency in cycles/pixel, mapped onto the
                    // profile's own circular grid.
                    let freq = x.min(patch_size - x) as f64 / patch_size as f64;
                    let idx = ((freq * n as f64).round() as usize).min(n / 2);
                    powers.push(profile[idx].max(F::zero()));
                }
                let mut sum = F::zero();
                for p in &powers {
                    sum += *p;
                }
                if sum > F::zero() {
                    let mean_power = sum / F::usize_as(patch_size);
                    powers.iter().map(|&p| (p / mean_power).sqrt()).collect()
                } else {
                    vec![F::one(); patch_size]
                }
            }
        }
    };

    let neg_half = F::from_f64_c(-0.5);
    for y in 0..patch_size {
        let freq_dist = y.min(patch_size - y);
        let freq_dist_f = F::usize_as(freq_dist);
        let normalized = freq_dist_f / psd_width;
        let value = (neg_half * normalized * normalized).exp();

        for x in 0..patch_size {
            sigma_psd[[y, x]] = value * x_gain[x];
        }
    }

    sigma_psd
}

fn subtract_streak_profile<F: Bm3dFloat>(
    image: &mut Array2<F>,
    streak_sigma_smooth: F,
    streak_iterations: usize,
) {
    let (rows, _cols) = image.dim();

    let profile =
        estimate_streak_profile_impl(image.view(), streak_sigma_smooth, streak_iterations);

    for r in 0..rows {
        let mut row = image.row_mut(r);
        for (c, val) in row.iter_mut().enumerate() {
            *val -= profile[c];
        }
    }
}

pub fn bm3d_ring_artifact_removal_with_plans<F: Bm3dFloat>(
    sinogram: ArrayView2<F>,
    mode: RingRemovalMode,
    config: &Bm3dConfig<F>,
    plans: &crate::pipeline::Bm3dPlans<F>,
) -> Result<Array2<F>, String> {
    bm3d_ring_artifact_removal_with_psd(sinogram, mode, config, plans, None)
}

pub(crate) fn bm3d_ring_artifact_removal_with_psd<F: Bm3dFloat>(
    sinogram: ArrayView2<F>,
    mode: RingRemovalMode,
    config: &Bm3dConfig<F>,
    plans: &crate::pipeline::Bm3dPlans<F>,
    psd_x: Option<ArrayView1<F>>,
) -> Result<Array2<F>, String> {
    config.validate()?;
    let profile_timing = profile_timing_enabled();
    let total_started = profile_timing.then(Instant::now);
    let mut normalize_ns = 0u128;
    let mut sigma_map_ns = 0u128;
    let mut psd_ns = 0u128;
    let mut prefilter_ns = 0u128;
    let mut sigma_estimate_ns = 0u128;
    let mut hard_pass_ns = 0u128;
    let mut wiener_pass_ns = 0u128;
    let mut denormalize_ns = 0u128;

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

    let (rows, cols) = sinogram.dim();

    check_image_size(rows, cols, config.patch_size)?;

    let (d_min, _d_max, range, eps, mut z_norm) = timed!(profile_timing, normalize_ns, {
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

        let z_norm = if range > eps {
            sinogram.mapv(|x| (x - d_min) / range)
        } else {
            Array2::zeros((rows, cols))
        };
        (d_min, d_max, range, eps, z_norm)
    });

    let sigma_map = timed!(profile_timing, sigma_map_ns, {
        compute_sigma_map(
            z_norm.view(),
            config.sigma_map_smoothing,
            config.streak_sigma_scale,
        )
    });

    let sigma_psd = timed!(profile_timing, psd_ns, {
        match mode {
            RingRemovalMode::Generic => {
                // Scalar PSD (no colored noise model)
                Array2::zeros((1, 1))
            }
            RingRemovalMode::Streak | RingRemovalMode::MultiscaleStreak => {
                construct_psd(config.patch_size, config.psd_width, psd_x)
            }
            RingRemovalMode::FourierSvd => Array2::zeros((1, 1)),
        }
    });

    timed!(profile_timing, prefilter_ns, {
        if mode == RingRemovalMode::Streak || mode == RingRemovalMode::MultiscaleStreak {
            subtract_streak_profile(
                &mut z_norm,
                config.streak_sigma_smooth,
                config.streak_iterations,
            );
        } else if mode == RingRemovalMode::FourierSvd {
            z_norm = crate::fourier_svd::fourier_svd_removal(
                z_norm.view(),
                config.fft_alpha,
                config.notch_width,
            );
        }
    });

    let sigma_random = timed!(profile_timing, sigma_estimate_ns, {
        if config.sigma_random <= F::from_f64_c(1e-6) {
            estimate_noise_sigma(z_norm.view())
        } else {
            config.sigma_random
        }
    });

    let search_step = config.resolved_search_step(mode);
    let hard_config = Bm3dKernelConfig {
        sigma_random,
        threshold: config.threshold,
        patch_size: config.patch_size,
        step_size: config.step_size,
        search_window: config.search_window,
        max_matches: config.max_matches,
        search_step,
        use_hadamard_fast_path: config.use_hadamard_fast_path,
    };
    let yhat_ht = timed!(profile_timing, hard_pass_ns, {
        run_bm3d_step(
            z_norm.view(),
            z_norm.view(),
            Bm3dMode::HardThreshold,
            sigma_psd.view(),
            sigma_map.view(),
            &hard_config,
            plans,
        )
    })?;

    let wiener_config = Bm3dKernelConfig {
        sigma_random,
        threshold: F::zero(), // not used for Wiener
        patch_size: config.patch_size,
        step_size: config.step_size,
        search_window: config.search_window,
        max_matches: config.max_matches,
        search_step,
        use_hadamard_fast_path: config.use_hadamard_fast_path,
    };
    let yhat_final = timed!(profile_timing, wiener_pass_ns, {
        run_bm3d_step(
            z_norm.view(),
            yhat_ht.view(),
            Bm3dMode::Wiener,
            sigma_psd.view(),
            sigma_map.view(),
            &wiener_config,
            plans,
        )
    })?;

    let output = timed!(profile_timing, denormalize_ns, {
        if range > eps {
            yhat_final.mapv(|x| x * range + d_min)
        } else {
            Array2::from_elem(yhat_final.raw_dim(), d_min)
        }
    });

    if profile_timing {
        let total_ms = total_started
            .map(|t| t.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or_default();
        eprintln!(
            "bm3d_orch_profile mode={:?} size={}x{} total_ms={:.3} normalize_ms={:.3} sigma_map_ms={:.3} psd_ms={:.3} prefilter_ms={:.3} sigma_est_ms={:.3} hard_ms={:.3} wiener_ms={:.3} denorm_ms={:.3}",
            mode,
            rows,
            cols,
            total_ms,
            normalize_ns as f64 / 1_000_000.0,
            sigma_map_ns as f64 / 1_000_000.0,
            psd_ns as f64 / 1_000_000.0,
            prefilter_ns as f64 / 1_000_000.0,
            sigma_estimate_ns as f64 / 1_000_000.0,
            hard_pass_ns as f64 / 1_000_000.0,
            wiener_pass_ns as f64 / 1_000_000.0,
            denormalize_ns as f64 / 1_000_000.0,
        );
    }

    Ok(output)
}

pub fn bm3d_ring_artifact_removal<F: Bm3dFloat>(
    sinogram: ArrayView2<F>,
    mode: RingRemovalMode,
    config: &Bm3dConfig<F>,
) -> Result<Array2<F>, String> {
    let plans = crate::pipeline::Bm3dPlans::new(config.patch_size, config.max_matches);
    bm3d_ring_artifact_removal_with_plans(sinogram, mode, config, &plans)
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

        // Box-Muller transform for a standard Gaussian sample
        fn next_gaussian(&mut self) -> f64 {
            let u1 = f64::from(self.next_f32()).max(1e-10);
            let u2 = f64::from(self.next_f32());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
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
    fn test_default_config_matches_spec() {
        let config: Bm3dConfig<f32> = Bm3dConfig::default();

        assert_eq!(RingRemovalMode::default(), RingRemovalMode::Streak);
        assert!(approx_eq(config.sigma_random, 0.1, 1e-6));
        assert_eq!(config.patch_size, 8);
        assert_eq!(config.step_size, 4);
        assert_eq!(config.search_window, 24);
        assert_eq!(config.max_matches, 16);
        assert_eq!(config.search_step, None);
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
        let config: Bm3dConfig<f32> = Bm3dConfig {
            patch_size: 0,
            ..Bm3dConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_step_size() {
        let config: Bm3dConfig<f32> = Bm3dConfig {
            step_size: 0,
            ..Bm3dConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_negative_sigma() {
        let config = Bm3dConfig {
            sigma_random: -0.1,
            ..Bm3dConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_psd_width() {
        let config = Bm3dConfig {
            psd_width: 0.0,
            ..Bm3dConfig::default()
        };
        assert!(config.validate().is_err());

        let config = Bm3dConfig {
            psd_width: -0.1,
            ..Bm3dConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_sigma_map_shape() {
        let image = random_matrix(64, 128, 12345);
        let normalized = image.mapv(|x| x); // Already in [0,1]

        let sigma_map = compute_sigma_map(normalized.view(), 20.0, 1.1);

        assert_eq!(sigma_map.dim(), (1, 128));
    }

    #[test]
    fn test_sigma_map_compact_row_profile() {
        let image = random_matrix(32, 64, 54321);
        let sigma_map = compute_sigma_map(image.view(), 20.0, 1.1);
        assert_eq!(sigma_map.nrows(), 1);
        assert_eq!(sigma_map.ncols(), 64);
    }

    #[test]
    fn test_sigma_map_non_negative() {
        let image = random_matrix(32, 64, 11111);
        let sigma_map = compute_sigma_map(image.view(), 20.0, 1.1);

        for &val in sigma_map.iter() {
            assert!(val >= 0.0, "Sigma map should be non-negative");
        }
    }

    #[test]
    fn test_psd_shape() {
        let psd = construct_psd::<f32>(8, 0.6, None);
        assert_eq!(psd.dim(), (8, 8));
    }

    #[test]
    fn test_psd_columns_identical() {
        let psd = construct_psd::<f32>(8, 0.6, None);

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
        let psd = construct_psd::<f32>(8, 0.6, None);
        assert!(approx_eq(psd[[0, 0]], 1.0, 1e-6));

        let nyquist = 4; // patch_size / 2
        for y in 1..=nyquist {
            assert!(
                psd[[y, 0]] < psd[[y - 1, 0]],
                "PSD should decrease from DC to Nyquist at y={}",
                y
            );
        }
        for y in (nyquist + 1)..8 {
            assert!(
                psd[[y, 0]] > psd[[y - 1, 0]],
                "PSD should increase from Nyquist back to DC at y={}",
                y
            );
        }
    }

    #[test]
    fn test_psd_all_positive() {
        let psd = construct_psd::<f32>(8, 0.6, None);
        for &val in psd.iter() {
            assert!(val > 0.0, "PSD values should be positive");
        }
    }

    #[test]
    fn test_psd_fft_symmetry() {
        for &patch_size in &[4usize, 8, 16] {
            let psd = construct_psd::<f32>(patch_size, 0.6, None);
            for y in 1..patch_size {
                let mirror = patch_size - y;
                assert!(
                    approx_eq(psd[[y, 0]], psd[[mirror, 0]], 1e-6),
                    "PSD symmetry broken: psd[{},0]={} != psd[{},0]={} for patch_size={}",
                    y,
                    psd[[y, 0]],
                    mirror,
                    psd[[mirror, 0]],
                    patch_size,
                );
            }
        }
    }

    #[test]
    fn test_psd_symmetry_known_values() {
        let psd = construct_psd::<f32>(8, 0.6, None);
        let expected_bin1 = (-0.5_f32 * (1.0_f32 / 0.6).powi(2)).exp();
        assert!(
            approx_eq(psd[[1, 0]], expected_bin1, 1e-6),
            "Bin 1: expected {} got {}",
            expected_bin1,
            psd[[1, 0]]
        );
        assert!(
            approx_eq(psd[[7, 0]], expected_bin1, 1e-6),
            "Bin 7 should equal Bin 1: expected {} got {}",
            expected_bin1,
            psd[[7, 0]]
        );
    }

    #[test]
    fn test_psd_odd_patch_size_symmetry() {
        for &patch_size in &[5usize, 7, 9] {
            let psd = construct_psd::<f32>(patch_size, 0.6, None);
            for y in 1..patch_size {
                let mirror = patch_size - y;
                assert!(
                    approx_eq(psd[[y, 0]], psd[[mirror, 0]], 1e-6),
                    "Odd patch_size={}: psd[{},0]={} != psd[{},0]={}",
                    patch_size,
                    y,
                    psd[[y, 0]],
                    mirror,
                    psd[[mirror, 0]],
                );
            }
        }
    }

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

    #[test]
    fn test_handles_non_normalized_input() {
        let image = Array2::from_shape_fn((32, 32), |(r, c)| 100.0 + (r * 32 + c) as f32 * 10.0);
        let config = Bm3dConfig::default();

        let result = bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Generic, &config);

        assert!(result.is_ok());
        let output = result.unwrap();

        let out_min = output.iter().copied().fold(f32::INFINITY, f32::min);
        let out_max = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let in_min = image.iter().copied().fold(f32::INFINITY, f32::min);
        let in_max = image.iter().copied().fold(f32::NEG_INFINITY, f32::max);

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
        let constant_val = 42.5f32;
        let image = Array2::from_elem((32, 32), constant_val);
        let config = Bm3dConfig::default();

        let result = bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Generic, &config);

        assert!(result.is_ok());
        let output = result.unwrap();

        for &val in output.iter() {
            assert!(
                approx_eq(val, constant_val, 1e-5),
                "Constant image should remain constant, got {}",
                val
            );
        }
    }

    #[test]
    fn test_generic_and_streak_differ() {
        let mut image = Array2::from_elem((32, 64), 0.5f32);
        for r in 0..32 {
            image[[r, 20]] = 0.9;
        }

        let config = Bm3dConfig::default();

        let result_generic =
            bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Generic, &config).unwrap();
        let result_streak =
            bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Streak, &config).unwrap();

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
        let mut image = Array2::from_elem((64, 64), 0.5f32);
        for r in 0..64 {
            image[[r, 32]] = 1.0;
        }

        let config = Bm3dConfig::default();

        let result =
            bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Streak, &config).unwrap();

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

        assert!(
            col_variance < orig_col_variance,
            "Streak mode should reduce column variance: {} >= {}",
            col_variance,
            orig_col_variance
        );
    }

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
        let config = Bm3dConfig {
            patch_size: 0,
            ..Bm3dConfig::default()
        };

        let result = bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Generic, &config);

        assert!(result.is_err());
    }

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

    #[test]
    fn test_different_patch_sizes() {
        let image = random_matrix(48, 48, 11111);

        for patch_size in [4, 8] {
            let config = Bm3dConfig {
                patch_size,
                step_size: patch_size / 2,
                ..Bm3dConfig::default()
            };

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
            let config = Bm3dConfig {
                sigma_random: sigma,
                ..Bm3dConfig::default()
            };

            let result =
                bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Generic, &config);

            assert!(result.is_ok(), "Failed for sigma={}", sigma);
        }
    }
    #[test]
    fn test_auto_sigma_estimation() {
        let mut rng = SimpleLcg::new(999);
        let image = Array2::from_shape_fn((64, 64), |_| rng.next_f32());

        let config = Bm3dConfig {
            sigma_random: 0.0,
            ..Bm3dConfig::default()
        };

        let result = bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Generic, &config);

        assert!(result.is_ok(), "Auto-estimation should succeed");
        let output = result.unwrap();
        assert_eq!(output.dim(), image.dim());

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

    const STRIP_PATCH_SIZE: usize = 8;
    const STRIP_STEP_SIZE: usize = 4;
    const STRIP_NOISE_SIGMA: f64 = 0.05;
    const STRIP_MAX_RESIDUAL_RATIO: f64 = 0.6;

    /// Smooth horizontal bands plus Gaussian noise, with a fixed seed per shape.
    fn banded_images<F: Bm3dFloat>(rows: usize, cols: usize) -> (Array2<F>, Array2<F>) {
        let clean =
            Array2::from_shape_fn((rows, cols), |(r, _)| 0.5 + 0.2 * (r as f64 / 15.0).sin());
        let mut rng = SimpleLcg::new((rows * 1000 + cols) as u64);
        let noisy = clean.mapv(|v| v + STRIP_NOISE_SIGMA * rng.next_gaussian());
        (clean.mapv(F::from_f64_c), noisy.mapv(F::from_f64_c))
    }

    fn edge_residual_report<F: Bm3dFloat>(
        cases: &[((usize, usize), usize, usize)],
        precision: &str,
    ) -> (Vec<String>, bool) {
        use ndarray::s;

        let mut lines = Vec::new();
        let mut any_failed = false;
        for &((rows, cols), strip_rows, strip_cols) in cases {
            let (clean, noisy) = banded_images::<F>(rows, cols);
            let lo = noisy.iter().fold(F::infinity(), |a, &b| a.min(b));
            let hi = noisy.iter().fold(F::neg_infinity(), |a, &b| a.max(b));
            let range = (hi - lo).to_f64().unwrap();
            // sigma_random applies to the min/max-normalised image.
            let config = Bm3dConfig {
                sigma_random: F::from_f64_c(STRIP_NOISE_SIGMA / range),
                patch_size: STRIP_PATCH_SIZE,
                step_size: STRIP_STEP_SIZE,
                ..Bm3dConfig::default()
            };
            for mode in [RingRemovalMode::Generic, RingRemovalMode::Streak] {
                let output = bm3d_ring_artifact_removal(noisy.view(), mode, &config).unwrap();
                let residual_ratio = |region: (usize, usize, usize, usize)| -> f64 {
                    let (r0, r1, c0, c1) = region;
                    let out = output.slice(s![r0..r1, c0..c1]);
                    let truth = clean.slice(s![r0..r1, c0..c1]);
                    let sum_sq: f64 = out
                        .iter()
                        .zip(truth.iter())
                        .map(|(&o, &t)| (o - t).to_f64().unwrap().powi(2))
                        .sum();
                    (sum_sq / out.len() as f64).sqrt() / STRIP_NOISE_SIGMA
                };
                let mut regions = Vec::new();
                if strip_rows > 0 {
                    regions.push(("bottom", (rows - strip_rows, rows, 0, cols)));
                }
                if strip_cols > 0 {
                    regions.push(("right", (0, rows, cols - strip_cols, cols)));
                }
                for (edge, region) in regions {
                    let ratio = residual_ratio(region);
                    let failed = ratio >= STRIP_MAX_RESIDUAL_RATIO;
                    any_failed |= failed;
                    lines.push(format!(
                        "{rows}x{cols} {mode:?} {precision} {edge} strip: residual \
                         {ratio:.3} sigma{}",
                        if failed { "  <-- unfiltered" } else { "" }
                    ));
                }
            }
        }
        (lines, any_failed)
    }

    #[test]
    fn test_public_2d_filters_bottom_and_right_strip() {
        // (extent - 8) % 4 is 1, 2 or 3 on rows, columns or both; the strip
        // is that many rows or columns wide.
        let cases: Vec<_> = [(66, 82), (65, 81), (67, 83), (64, 83), (67, 80)]
            .into_iter()
            .map(|(rows, cols)| {
                (
                    (rows, cols),
                    (rows - STRIP_PATCH_SIZE) % STRIP_STEP_SIZE,
                    (cols - STRIP_PATCH_SIZE) % STRIP_STEP_SIZE,
                )
            })
            .collect();
        let (mut lines, f32_failed) = edge_residual_report::<f32>(&cases, "f32");
        let (f64_lines, f64_failed) = edge_residual_report::<f64>(&cases, "f64");
        lines.extend(f64_lines);
        assert!(
            !(f32_failed || f64_failed),
            "edge strip keeps the input noise (bound {STRIP_MAX_RESIDUAL_RATIO} sigma):\n{}",
            lines.join("\n")
        );
    }

    #[test]
    fn test_public_2d_filters_edges_of_aligned_shape() {
        // Control: on 64 x 80 the plain grid reaches both edges, so its last
        // two rows and columns are filtered.
        let cases = [((64, 80), 2, 2)];
        let (mut lines, f32_failed) = edge_residual_report::<f32>(&cases, "f32");
        let (f64_lines, f64_failed) = edge_residual_report::<f64>(&cases, "f64");
        lines.extend(f64_lines);
        assert!(
            !(f32_failed || f64_failed),
            "edge rows or columns keep the input noise (bound {STRIP_MAX_RESIDUAL_RATIO} \
             sigma):\n{}",
            lines.join("\n")
        );
    }
}
