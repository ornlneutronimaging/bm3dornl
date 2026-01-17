//! PyO3 Python bindings for BM3D denoising.
//!
//! This crate provides thin Python bindings for the bm3d_core library.
//! All algorithm logic is in bm3d_core; this crate only handles
//! Python/NumPy type conversions.
//!
//! Both f32 and f64 precision are supported. Functions with `_f64` suffix
//! accept and return float64 numpy arrays.

use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;

use bm3d_core::{bm3d_ring_artifact_removal, Bm3dConfig, RingRemovalMode};
use bm3d_core::{multiscale_bm3d_streak_removal, MultiscaleConfig};
use bm3d_core::{run_bm3d_step, run_bm3d_step_stack, Bm3dMode};

/// Hard thresholding step of BM3D for a single 2D image.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn bm3d_hard_thresholding<'py>(
    py: Python<'py>,
    input_noisy: PyReadonlyArray2<'py, f32>,
    input_pilot: PyReadonlyArray2<'py, f32>,
    sigma_psd: PyReadonlyArray2<'py, f32>,
    sigma_map: PyReadonlyArray2<'py, f32>,
    sigma_random: f32,
    threshold: f32,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let plans = bm3d_core::pipeline::Bm3dPlans::new(patch_size, max_matches);
    let output = run_bm3d_step(
        input_noisy.as_array(),
        input_pilot.as_array(),
        Bm3dMode::HardThreshold,
        sigma_psd.as_array(),
        sigma_map.as_array(),
        sigma_random,
        threshold,
        patch_size,
        step_size,
        search_window,
        max_matches,
        &plans,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(output.to_pyarray(py))
}

/// Wiener filtering step of BM3D for a single 2D image.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn bm3d_wiener_filtering<'py>(
    py: Python<'py>,
    input_noisy: PyReadonlyArray2<'py, f32>,
    input_pilot: PyReadonlyArray2<'py, f32>,
    sigma_psd: PyReadonlyArray2<'py, f32>,
    sigma_map: PyReadonlyArray2<'py, f32>,
    sigma_random: f32,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let plans = bm3d_core::pipeline::Bm3dPlans::new(patch_size, max_matches);
    let output = run_bm3d_step(
        input_noisy.as_array(),
        input_pilot.as_array(),
        Bm3dMode::Wiener,
        sigma_psd.as_array(),
        sigma_map.as_array(),
        sigma_random,
        0.0,
        patch_size,
        step_size,
        search_window,
        max_matches,
        &plans,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(output.to_pyarray(py))
}

/// Hard thresholding step of BM3D for a 3D stack of images.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn bm3d_hard_thresholding_stack<'py>(
    py: Python<'py>,
    input_noisy: PyReadonlyArray3<'py, f32>,
    input_pilot: PyReadonlyArray3<'py, f32>,
    sigma_psd: PyReadonlyArray2<'py, f32>,
    sigma_map: PyReadonlyArray3<'py, f32>,
    sigma_random: f32,
    threshold: f32,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let plans = bm3d_core::pipeline::Bm3dPlans::new(patch_size, max_matches);
    let output = run_bm3d_step_stack(
        input_noisy.as_array(),
        input_pilot.as_array(),
        Bm3dMode::HardThreshold,
        sigma_psd.as_array(),
        sigma_map.as_array(),
        sigma_random,
        threshold,
        patch_size,
        step_size,
        search_window,
        max_matches,
        &plans,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(output.to_pyarray(py))
}

/// Wiener filtering step of BM3D for a 3D stack of images.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn bm3d_wiener_filtering_stack<'py>(
    py: Python<'py>,
    input_noisy: PyReadonlyArray3<'py, f32>,
    input_pilot: PyReadonlyArray3<'py, f32>,
    sigma_psd: PyReadonlyArray2<'py, f32>,
    sigma_map: PyReadonlyArray3<'py, f32>,
    sigma_random: f32,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let plans = bm3d_core::pipeline::Bm3dPlans::new(patch_size, max_matches);
    let output = run_bm3d_step_stack(
        input_noisy.as_array(),
        input_pilot.as_array(),
        Bm3dMode::Wiener,
        sigma_psd.as_array(),
        sigma_map.as_array(),
        sigma_random,
        0.0,
        patch_size,
        step_size,
        search_window,
        max_matches,
        &plans,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(output.to_pyarray(py))
}

/// Test function for block matching (used for debugging/validation).
#[pyfunction]
pub fn test_block_matching_rust(
    input: PyReadonlyArray2<'_, f32>,
    ref_r: usize,
    ref_c: usize,
    patch_size: usize,
    search_win: usize,
    max_matches: usize,
) -> PyResult<Vec<(usize, usize, f32)>> {
    let result = bm3d_core::pipeline::test_block_matching(
        input.as_array(),
        ref_r,
        ref_c,
        patch_size,
        search_win,
        max_matches,
    );
    Ok(result)
}

/// Estimate the static vertical streak profile using an iterative robust approach.
#[pyfunction]
#[pyo3(name = "estimate_streak_profile_rust")]
pub fn estimate_streak_profile_py<'py>(
    py: Python<'py>,
    sinogram: PyReadonlyArray2<'py, f32>,
    sigma_smooth: f32,
    iterations: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let sinogram_view = sinogram.as_array();
    let result = bm3d_core::estimate_streak_profile_impl(sinogram_view, sigma_smooth, iterations);
    Ok(result.to_pyarray(py))
}

// ============================================================================
// f64 (double precision) variants
// ============================================================================

/// Hard thresholding step of BM3D for a single 2D image (f64 precision).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn bm3d_hard_thresholding_f64<'py>(
    py: Python<'py>,
    input_noisy: PyReadonlyArray2<'py, f64>,
    input_pilot: PyReadonlyArray2<'py, f64>,
    sigma_psd: PyReadonlyArray2<'py, f64>,
    sigma_map: PyReadonlyArray2<'py, f64>,
    sigma_random: f64,
    threshold: f64,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let plans = bm3d_core::pipeline::Bm3dPlans::new(patch_size, max_matches);
    let output = run_bm3d_step(
        input_noisy.as_array(),
        input_pilot.as_array(),
        Bm3dMode::HardThreshold,
        sigma_psd.as_array(),
        sigma_map.as_array(),
        sigma_random,
        threshold,
        patch_size,
        step_size,
        search_window,
        max_matches,
        &plans,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(output.to_pyarray(py))
}

/// Wiener filtering step of BM3D for a single 2D image (f64 precision).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn bm3d_wiener_filtering_f64<'py>(
    py: Python<'py>,
    input_noisy: PyReadonlyArray2<'py, f64>,
    input_pilot: PyReadonlyArray2<'py, f64>,
    sigma_psd: PyReadonlyArray2<'py, f64>,
    sigma_map: PyReadonlyArray2<'py, f64>,
    sigma_random: f64,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let plans = bm3d_core::pipeline::Bm3dPlans::new(patch_size, max_matches);
    let output = run_bm3d_step(
        input_noisy.as_array(),
        input_pilot.as_array(),
        Bm3dMode::Wiener,
        sigma_psd.as_array(),
        sigma_map.as_array(),
        sigma_random,
        0.0,
        patch_size,
        step_size,
        search_window,
        max_matches,
        &plans,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(output.to_pyarray(py))
}

/// Hard thresholding step of BM3D for a 3D stack of images (f64 precision).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn bm3d_hard_thresholding_stack_f64<'py>(
    py: Python<'py>,
    input_noisy: PyReadonlyArray3<'py, f64>,
    input_pilot: PyReadonlyArray3<'py, f64>,
    sigma_psd: PyReadonlyArray2<'py, f64>,
    sigma_map: PyReadonlyArray3<'py, f64>,
    sigma_random: f64,
    threshold: f64,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let plans = bm3d_core::pipeline::Bm3dPlans::new(patch_size, max_matches);
    let output = run_bm3d_step_stack(
        input_noisy.as_array(),
        input_pilot.as_array(),
        Bm3dMode::HardThreshold,
        sigma_psd.as_array(),
        sigma_map.as_array(),
        sigma_random,
        threshold,
        patch_size,
        step_size,
        search_window,
        max_matches,
        &plans,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(output.to_pyarray(py))
}

/// Wiener filtering step of BM3D for a 3D stack of images (f64 precision).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn bm3d_wiener_filtering_stack_f64<'py>(
    py: Python<'py>,
    input_noisy: PyReadonlyArray3<'py, f64>,
    input_pilot: PyReadonlyArray3<'py, f64>,
    sigma_psd: PyReadonlyArray2<'py, f64>,
    sigma_map: PyReadonlyArray3<'py, f64>,
    sigma_random: f64,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let plans = bm3d_core::pipeline::Bm3dPlans::new(patch_size, max_matches);
    let output = run_bm3d_step_stack(
        input_noisy.as_array(),
        input_pilot.as_array(),
        Bm3dMode::Wiener,
        sigma_psd.as_array(),
        sigma_map.as_array(),
        sigma_random,
        0.0,
        patch_size,
        step_size,
        search_window,
        max_matches,
        &plans,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(output.to_pyarray(py))
}

/// Test function for block matching (f64 precision).
#[pyfunction]
pub fn test_block_matching_rust_f64(
    input: PyReadonlyArray2<'_, f64>,
    ref_r: usize,
    ref_c: usize,
    patch_size: usize,
    search_win: usize,
    max_matches: usize,
) -> PyResult<Vec<(usize, usize, f64)>> {
    let result = bm3d_core::pipeline::test_block_matching(
        input.as_array(),
        ref_r,
        ref_c,
        patch_size,
        search_win,
        max_matches,
    );
    Ok(result)
}

/// Estimate the static vertical streak profile (f64 precision).
#[pyfunction]
#[pyo3(name = "estimate_streak_profile_rust_f64")]
pub fn estimate_streak_profile_py_f64<'py>(
    py: Python<'py>,
    sinogram: PyReadonlyArray2<'py, f64>,
    sigma_smooth: f64,
    iterations: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let sinogram_view = sinogram.as_array();
    let result = bm3d_core::estimate_streak_profile_impl(sinogram_view, sigma_smooth, iterations);
    Ok(result.to_pyarray(py))
}

// ============================================================================
// Unified BM3D Ring Artifact Removal (complete pipeline)
// ============================================================================

/// Unified BM3D ring artifact removal for 2D sinograms (f32 precision).
///
/// This function performs the complete BM3D denoising pipeline:
/// 1. Normalizes input to [0, 1] range
/// 2. Computes spatially adaptive sigma map
/// 3. (Streak mode) Subtracts estimated streak profile
/// 4. Runs two-pass BM3D: Hard Threshold → Wiener
/// 5. Denormalizes output to original range
///
/// Parameters
/// ----------
/// sinogram : numpy.ndarray
///     Input 2D sinogram (H × W), dtype float32.
/// mode : str
///     Processing mode: "generic", "streak", "multiscale_streak", or "fourier_svd".
/// sigma_random : float, optional
///     Random noise std dev. Default: 0.0 (auto-estimate)
/// patch_size : int, optional
///     Block matching patch size. Default: 8
/// step_size : int, optional
///     Stride between patches. Default: 4
/// search_window : int, optional
///     Search window size for block matching. Default: 24
/// max_matches : int, optional
///     Maximum similar patches per group. Default: 16
/// threshold : float, optional
///     Hard thresholding coefficient. Default: 2.7
/// streak_sigma_smooth : float, optional
///     Sigma for streak estimation smoothing (Streak Mode). Default: 3.0
/// streak_iterations : int, optional
///     Number of streak estimation iterations (Streak Mode). Default: 2
/// sigma_map_smoothing : float, optional
///     Sigma for sigma map profile smoothing. Default: 20.0
/// streak_sigma_scale : float, optional
///     Sigma map scaling factor. Default: 1.1
/// psd_width : float, optional
///     PSD Gaussian width for streak mode. Default: 0.6
/// fft_alpha : float, optional
///     FFT-Guided SVD Trust Factor. Default: 1.0 (Fourier-SVD mode only)
/// notch_width : float, optional
///     FFT-Guided SVD Notch Width. Default: 2.0 (Fourier-SVD mode only)
///
/// Returns
/// -------
/// numpy.ndarray
///     Denoised sinogram with same shape as input.
#[pyfunction]
#[pyo3(signature = (
    sinogram,
    mode,
    sigma_random = None,
    patch_size = None,
    step_size = None,
    search_window = None,
    max_matches = None,
    threshold = None,
    streak_sigma_smooth = None,
    streak_iterations = None,
    sigma_map_smoothing = None,
    streak_sigma_scale = None,
    psd_width = None,
    fft_alpha = None,
    notch_width = None
))]
#[allow(clippy::too_many_arguments)]
pub fn bm3d_ring_artifact_removal_2d<'py>(
    py: Python<'py>,
    sinogram: PyReadonlyArray2<'py, f32>,
    mode: &str,
    sigma_random: Option<f32>,
    patch_size: Option<usize>,
    step_size: Option<usize>,
    search_window: Option<usize>,
    max_matches: Option<usize>,
    threshold: Option<f32>,
    streak_sigma_smooth: Option<f32>,
    streak_iterations: Option<usize>,
    sigma_map_smoothing: Option<f32>,
    streak_sigma_scale: Option<f32>,
    psd_width: Option<f32>,
    fft_alpha: Option<f32>,
    notch_width: Option<f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    // Parse mode
    let ring_mode = match mode.to_lowercase().as_str() {
        "generic" => RingRemovalMode::Generic,
        "streak" => RingRemovalMode::Streak,
        "multiscale_streak" | "multiscalestreak" => RingRemovalMode::MultiscaleStreak,
        "fourier_svd" | "fouriersvd" => RingRemovalMode::FourierSvd,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid mode '{}'. Expected 'generic', 'streak', 'multiscale_streak', or 'fourier_svd'.",
                mode
            )))
        }
    };

    // Build config with defaults
    let mut config: Bm3dConfig<f32> = Bm3dConfig::default();

    if let Some(v) = sigma_random {
        config.sigma_random = v;
    }
    if let Some(v) = patch_size {
        config.patch_size = v;
    }
    if let Some(v) = step_size {
        config.step_size = v;
    }
    if let Some(v) = search_window {
        config.search_window = v;
    }
    if let Some(v) = max_matches {
        config.max_matches = v;
    }
    if let Some(v) = threshold {
        config.threshold = v;
    }
    if let Some(v) = streak_sigma_smooth {
        config.streak_sigma_smooth = v;
    }
    if let Some(v) = streak_iterations {
        config.streak_iterations = v;
    }
    if let Some(v) = sigma_map_smoothing {
        config.sigma_map_smoothing = v;
    }
    if let Some(v) = streak_sigma_scale {
        config.streak_sigma_scale = v;
    }
    if let Some(v) = psd_width {
        config.psd_width = v;
    }
    if let Some(v) = fft_alpha {
        config.fft_alpha = v;
    }
    if let Some(v) = notch_width {
        config.notch_width = v;
    }

    // Call Rust implementation
    let output = bm3d_ring_artifact_removal(sinogram.as_array(), ring_mode, &config)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok(output.to_pyarray(py))
}

/// Unified BM3D ring artifact removal for 2D sinograms (f64 precision).
///
/// See `bm3d_ring_artifact_removal_2d` for full documentation.
#[pyfunction]
#[pyo3(signature = (
    sinogram,
    mode,
    sigma_random = None,
    patch_size = None,
    step_size = None,
    search_window = None,
    max_matches = None,
    threshold = None,
    streak_sigma_smooth = None,
    streak_iterations = None,
    sigma_map_smoothing = None,
    streak_sigma_scale = None,
    psd_width = None,
    fft_alpha = None,
    notch_width = None
))]
#[allow(clippy::too_many_arguments)]
pub fn bm3d_ring_artifact_removal_2d_f64<'py>(
    py: Python<'py>,
    sinogram: PyReadonlyArray2<'py, f64>,
    mode: &str,
    sigma_random: Option<f64>,
    patch_size: Option<usize>,
    step_size: Option<usize>,
    search_window: Option<usize>,
    max_matches: Option<usize>,
    threshold: Option<f64>,
    streak_sigma_smooth: Option<f64>,
    streak_iterations: Option<usize>,
    sigma_map_smoothing: Option<f64>,
    streak_sigma_scale: Option<f64>,
    psd_width: Option<f64>,
    fft_alpha: Option<f64>,
    notch_width: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    // Parse mode
    let ring_mode = match mode.to_lowercase().as_str() {
        "generic" => RingRemovalMode::Generic,
        "streak" => RingRemovalMode::Streak,
        "multiscale_streak" | "multiscalestreak" => RingRemovalMode::MultiscaleStreak,
        "fourier_svd" | "fouriersvd" => RingRemovalMode::FourierSvd,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid mode '{}'. Expected 'generic', 'streak', 'multiscale_streak', or 'fourier_svd'.",
                mode
            )))
        }
    };

    // Build config with defaults
    let mut config: Bm3dConfig<f64> = Bm3dConfig::default();

    if let Some(v) = sigma_random {
        config.sigma_random = v;
    }
    if let Some(v) = patch_size {
        config.patch_size = v;
    }
    if let Some(v) = step_size {
        config.step_size = v;
    }
    if let Some(v) = search_window {
        config.search_window = v;
    }
    if let Some(v) = max_matches {
        config.max_matches = v;
    }
    if let Some(v) = threshold {
        config.threshold = v;
    }
    if let Some(v) = streak_sigma_smooth {
        config.streak_sigma_smooth = v;
    }
    if let Some(v) = streak_iterations {
        config.streak_iterations = v;
    }
    if let Some(v) = sigma_map_smoothing {
        config.sigma_map_smoothing = v;
    }
    if let Some(v) = streak_sigma_scale {
        config.streak_sigma_scale = v;
    }
    if let Some(v) = psd_width {
        config.psd_width = v;
    }
    if let Some(v) = fft_alpha {
        config.fft_alpha = v;
    }
    if let Some(v) = notch_width {
        config.notch_width = v;
    }

    // Call Rust implementation
    let output = bm3d_ring_artifact_removal(sinogram.as_array(), ring_mode, &config)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok(output.to_pyarray(py))
}

/// Fourier-SVD Streak Removal (Standalone) - f32
#[pyfunction]
#[pyo3(name = "fourier_svd_removal_rust")]
#[pyo3(signature = (sinogram, fft_alpha=1.0, notch_width=2.0))]
pub fn fourier_svd_removal_py<'py>(
    py: Python<'py>,
    sinogram: PyReadonlyArray2<'py, f32>,
    fft_alpha: f32,
    notch_width: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let output =
        bm3d_core::fourier_svd::fourier_svd_removal(sinogram.as_array(), fft_alpha, notch_width);
    Ok(output.to_pyarray(py))
}

/// Fourier-SVD Streak Removal (Standalone) - f64
#[pyfunction]
#[pyo3(name = "fourier_svd_removal_rust_f64")]
#[pyo3(signature = (sinogram, fft_alpha=1.0, notch_width=2.0))]
pub fn fourier_svd_removal_py_f64<'py>(
    py: Python<'py>,
    sinogram: PyReadonlyArray2<'py, f64>,
    fft_alpha: f64,
    notch_width: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let output =
        bm3d_core::fourier_svd::fourier_svd_removal(sinogram.as_array(), fft_alpha, notch_width);
    Ok(output.to_pyarray(py))
}

// ============================================================================
// Noise Estimation
// ============================================================================

/// Estimate noise standard deviation using MAD-based robust estimation (f32).
///
/// This implements sigma estimation from Mäkinen et al. (2021), optimized for
/// detecting vertical streak noise in sinograms. The image is filtered to isolate
/// vertical streaks (vertical Gaussian + horizontal High-pass), then MAD (Median
/// Absolute Deviation) is computed and scaled.
///
/// This is primarily a diagnostic tool for advanced users who want to understand
/// the noise characteristics of their data or tune denoising parameters.
///
/// Parameters
/// ----------
/// sinogram : numpy.ndarray
///     Input 2D sinogram (H × W), dtype float32.
///
/// Returns
/// -------
/// float
///     Estimated noise standard deviation (sigma).
#[pyfunction]
#[pyo3(name = "estimate_noise_sigma_rust")]
pub fn estimate_noise_sigma_py(sinogram: PyReadonlyArray2<'_, f32>) -> PyResult<f32> {
    Ok(bm3d_core::estimate_noise_sigma(sinogram.as_array()))
}

/// Estimate noise standard deviation using MAD-based robust estimation (f64).
///
/// See `estimate_noise_sigma_rust` for full documentation.
#[pyfunction]
#[pyo3(name = "estimate_noise_sigma_rust_f64")]
pub fn estimate_noise_sigma_py_f64(sinogram: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
    Ok(bm3d_core::estimate_noise_sigma(sinogram.as_array()))
}

// ============================================================================
// Multi-Scale BM3D Streak Removal
// ============================================================================

/// Multi-scale BM3D streak removal for 2D sinograms (f32 precision).
///
/// This function wraps single-scale BM3D in a coarse-to-fine pyramid
/// for handling wide streaks that single-scale cannot capture.
///
/// Parameters
/// ----------
/// sinogram : numpy.ndarray
///     Input 2D sinogram (H × W), dtype float32. Assumes log-transformed data.
/// num_scales : int, optional
///     Override automatic scale calculation. If None, uses floor(log2(width/40)).
/// filter_strength : float, optional
///     Multiplier for BM3D filtering intensity. Default: 1.0
/// threshold : float, optional
///     Hard threshold coefficient. Default: 3.5 (note: different from single-scale 2.7)
/// debin_iterations : int, optional
///     Iterations for cubic spline debinning. Default: 30
/// patch_size : int, optional
///     Block matching patch size. Default: 8
/// step_size : int, optional
///     Stride between patches. Default: 4
/// search_window : int, optional
///     Search window size for block matching. Default: 24
/// max_matches : int, optional
///     Maximum similar patches per group. Default: 16
///
/// Returns
/// -------
/// numpy.ndarray
///     Denoised sinogram with same shape as input.
#[pyfunction]
#[pyo3(signature = (
    sinogram,
    num_scales = None,
    filter_strength = None,
    threshold = None,
    debin_iterations = None,
    patch_size = None,
    step_size = None,
    search_window = None,
    max_matches = None,
    sigma_random = None,
    streak_sigma_smooth = None,
    streak_iterations = None,
    sigma_map_smoothing = None,
    streak_sigma_scale = None,
    psd_width = None
))]
#[allow(clippy::too_many_arguments)]
pub fn multiscale_bm3d_streak_removal_2d<'py>(
    py: Python<'py>,
    sinogram: PyReadonlyArray2<'py, f32>,
    num_scales: Option<usize>,
    filter_strength: Option<f32>,
    threshold: Option<f32>,
    debin_iterations: Option<usize>,
    patch_size: Option<usize>,
    step_size: Option<usize>,
    search_window: Option<usize>,
    max_matches: Option<usize>,
    sigma_random: Option<f32>,
    streak_sigma_smooth: Option<f32>,
    streak_iterations: Option<usize>,
    sigma_map_smoothing: Option<f32>,
    streak_sigma_scale: Option<f32>,
    psd_width: Option<f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    // Build config with defaults, using struct init syntax
    let default = MultiscaleConfig::<f32>::default();
    let mut config = MultiscaleConfig {
        num_scales,
        filter_strength: filter_strength.unwrap_or(default.filter_strength),
        threshold: threshold.unwrap_or(default.threshold),
        debin_iterations: debin_iterations.unwrap_or(default.debin_iterations),
        bm3d_config: default.bm3d_config,
    };

    // Also set inner BM3D threshold if provided
    if let Some(v) = threshold {
        config.bm3d_config.threshold = v;
    }

    // Propagate standard BM3D config
    if let Some(v) = patch_size {
        config.bm3d_config.patch_size = v;
    }
    if let Some(v) = step_size {
        config.bm3d_config.step_size = v;
    }
    if let Some(v) = search_window {
        config.bm3d_config.search_window = v;
    }
    if let Some(v) = max_matches {
        config.bm3d_config.max_matches = v;
    }
    if let Some(v) = sigma_random {
        config.bm3d_config.sigma_random = v;
    }
    if let Some(v) = streak_sigma_smooth {
        config.bm3d_config.streak_sigma_smooth = v;
    }
    if let Some(v) = streak_iterations {
        config.bm3d_config.streak_iterations = v;
    }
    if let Some(v) = sigma_map_smoothing {
        config.bm3d_config.sigma_map_smoothing = v;
    }
    if let Some(v) = streak_sigma_scale {
        config.bm3d_config.streak_sigma_scale = v;
    }
    if let Some(v) = psd_width {
        config.bm3d_config.psd_width = v;
    }

    // Call Rust implementation
    let output = multiscale_bm3d_streak_removal(sinogram.as_array(), &config)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok(output.to_pyarray(py))
}

/// Multi-scale BM3D streak removal for 2D sinograms (f64 precision).
///
/// See `multiscale_bm3d_streak_removal_2d` for full documentation.
#[pyfunction]
#[pyo3(signature = (
    sinogram,
    num_scales = None,
    filter_strength = None,
    threshold = None,
    debin_iterations = None,
    patch_size = None,
    step_size = None,
    search_window = None,
    max_matches = None,
    sigma_random = None,
    streak_sigma_smooth = None,
    streak_iterations = None,
    sigma_map_smoothing = None,
    streak_sigma_scale = None,
    psd_width = None
))]
#[allow(clippy::too_many_arguments)]
pub fn multiscale_bm3d_streak_removal_2d_f64<'py>(
    py: Python<'py>,
    sinogram: PyReadonlyArray2<'py, f64>,
    num_scales: Option<usize>,
    filter_strength: Option<f64>,
    threshold: Option<f64>,
    debin_iterations: Option<usize>,
    patch_size: Option<usize>,
    step_size: Option<usize>,
    search_window: Option<usize>,
    max_matches: Option<usize>,
    sigma_random: Option<f64>,
    streak_sigma_smooth: Option<f64>,
    streak_iterations: Option<usize>,
    sigma_map_smoothing: Option<f64>,
    streak_sigma_scale: Option<f64>,
    psd_width: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    // Build config with defaults, using struct init syntax
    let default = MultiscaleConfig::<f64>::default();
    let mut config = MultiscaleConfig {
        num_scales,
        filter_strength: filter_strength.unwrap_or(default.filter_strength),
        threshold: threshold.unwrap_or(default.threshold),
        debin_iterations: debin_iterations.unwrap_or(default.debin_iterations),
        bm3d_config: default.bm3d_config,
    };

    // Also set inner BM3D threshold if provided
    if let Some(v) = threshold {
        config.bm3d_config.threshold = v;
    }
    if let Some(v) = patch_size {
        config.bm3d_config.patch_size = v;
    }
    if let Some(v) = step_size {
        config.bm3d_config.step_size = v;
    }
    if let Some(v) = search_window {
        config.bm3d_config.search_window = v;
    }
    if let Some(v) = max_matches {
        config.bm3d_config.max_matches = v;
    }
    if let Some(v) = sigma_random {
        config.bm3d_config.sigma_random = v;
    }
    if let Some(v) = streak_sigma_smooth {
        config.bm3d_config.streak_sigma_smooth = v;
    }
    if let Some(v) = streak_iterations {
        config.bm3d_config.streak_iterations = v;
    }
    if let Some(v) = sigma_map_smoothing {
        config.bm3d_config.sigma_map_smoothing = v;
    }
    if let Some(v) = streak_sigma_scale {
        config.bm3d_config.streak_sigma_scale = v;
    }
    if let Some(v) = psd_width {
        config.bm3d_config.psd_width = v;
    }

    // Call Rust implementation
    let output = multiscale_bm3d_streak_removal(sinogram.as_array(), &config)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok(output.to_pyarray(py))
}

/// BM3D Rust accelerator module
#[pymodule]
fn bm3d_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // f32 (single precision) functions
    m.add_function(wrap_pyfunction!(bm3d_hard_thresholding, m)?)?;
    m.add_function(wrap_pyfunction!(bm3d_wiener_filtering, m)?)?;
    m.add_function(wrap_pyfunction!(bm3d_hard_thresholding_stack, m)?)?;
    m.add_function(wrap_pyfunction!(bm3d_wiener_filtering_stack, m)?)?;
    m.add_function(wrap_pyfunction!(test_block_matching_rust, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_streak_profile_py, m)?)?;
    m.add_function(wrap_pyfunction!(bm3d_ring_artifact_removal_2d, m)?)?;
    m.add_function(wrap_pyfunction!(multiscale_bm3d_streak_removal_2d, m)?)?;
    m.add_function(wrap_pyfunction!(fourier_svd_removal_py, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_noise_sigma_py, m)?)?;

    // f64 (double precision) functions
    m.add_function(wrap_pyfunction!(bm3d_hard_thresholding_f64, m)?)?;
    m.add_function(wrap_pyfunction!(bm3d_wiener_filtering_f64, m)?)?;
    m.add_function(wrap_pyfunction!(bm3d_hard_thresholding_stack_f64, m)?)?;
    m.add_function(wrap_pyfunction!(bm3d_wiener_filtering_stack_f64, m)?)?;
    m.add_function(wrap_pyfunction!(test_block_matching_rust_f64, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_streak_profile_py_f64, m)?)?;
    m.add_function(wrap_pyfunction!(bm3d_ring_artifact_removal_2d_f64, m)?)?;
    m.add_function(wrap_pyfunction!(multiscale_bm3d_streak_removal_2d_f64, m)?)?;
    m.add_function(wrap_pyfunction!(fourier_svd_removal_py_f64, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_noise_sigma_py_f64, m)?)?;

    Ok(())
}
