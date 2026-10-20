//! PyO3 Python bindings for BM3D denoising.
//!
//! Both f32 and f64 precision are supported.

use ndarray::{Array2, Array3, ArrayView, ArrayView2, ArrayView3, CowArray, Dimension};
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;

use bm3d_core::pipeline::check_image_size;
use bm3d_core::{Bm3dConfig, RingRemovalMode, bm3d_ring_artifact_removal};
use bm3d_core::{Bm3dFloat, Bm3dKernelConfig, Bm3dPlans};
use bm3d_core::{Bm3dMode, run_bm3d_step, run_bm3d_step_stack};
use bm3d_core::{MultiscaleConfig, multiscale_bm3d_streak_removal};

type ProgressFn = Box<dyn Fn(usize, usize) -> Result<(), String>>;

/// Sentinel prefix for KeyboardInterrupt errors propagated from Python callbacks.
const KEYBOARD_INTERRUPT_SENTINEL: &str = "KeyboardInterrupt:";

fn make_progress_fn(cb: Py<PyAny>) -> ProgressFn {
    Box::new(move |current: usize, total: usize| -> Result<(), String> {
        Python::attach(|py| {
            cb.call(py, (current, total), None)
                .map(|_| ())
                .map_err(|e| {
                    if e.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py) {
                        format!("{KEYBOARD_INTERRUPT_SENTINEL} {e}")
                    } else {
                        format!("Progress callback error: {e}")
                    }
                })
        })
    })
}

fn map_stack_error<T>(result: Result<T, String>) -> PyResult<T> {
    result.map_err(|e| {
        if e.starts_with(KEYBOARD_INTERRUPT_SENTINEL) {
            pyo3::exceptions::PyKeyboardInterrupt::new_err(e)
        } else {
            pyo3::exceptions::PyValueError::new_err(e)
        }
    })
}

fn standard_layout<A: Clone, D: Dimension>(array: ArrayView<'_, A, D>) -> CowArray<'_, A, D> {
    if array.is_standard_layout() {
        CowArray::from(array)
    } else {
        CowArray::from(array.as_standard_layout().into_owned())
    }
}

fn kernel_search_step(step_size: usize, search_step: Option<usize>) -> Result<usize, String> {
    match search_step {
        Some(0) => Err("search_step must be > 0 (or None for step_size)".to_string()),
        Some(step) => Ok(step),
        None => Ok(step_size),
    }
}

fn run_bm3d_step_compat<F: Bm3dFloat>(
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
    search_step: Option<usize>,
    plans: &Bm3dPlans<F>,
) -> Result<Array2<F>, String> {
    let config = Bm3dKernelConfig {
        sigma_random,
        threshold,
        patch_size,
        step_size,
        search_window,
        max_matches,
        search_step: kernel_search_step(step_size, search_step)?,
        use_hadamard_fast_path: None,
    };
    run_bm3d_step(
        input_noisy,
        input_pilot,
        mode,
        sigma_psd,
        sigma_map,
        &config,
        plans,
    )
}

fn run_bm3d_step_stack_compat<F: Bm3dFloat>(
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
    search_step: Option<usize>,
    plans: &Bm3dPlans<F>,
    progress_fn: Option<&dyn Fn(usize, usize) -> Result<(), String>>,
) -> Result<Array3<F>, String> {
    let config = Bm3dKernelConfig {
        sigma_random,
        threshold,
        patch_size,
        step_size,
        search_window,
        max_matches,
        search_step: kernel_search_step(step_size, search_step)?,
        use_hadamard_fast_path: None,
    };
    run_bm3d_step_stack(
        input_noisy,
        input_pilot,
        mode,
        sigma_psd,
        sigma_map,
        &config,
        plans,
        progress_fn,
    )
}

/// Hard thresholding step of BM3D for a single 2D image.
///
/// `search_step` is the candidate-search stride; `None` uses `step_size`.
#[pyfunction]
#[pyo3(signature = (input_noisy, input_pilot, sigma_psd, sigma_map, sigma_random, threshold, patch_size, step_size, search_window, max_matches, search_step=None))]
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
    search_step: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let plans = bm3d_core::pipeline::Bm3dPlans::new(patch_size, max_matches);
    let output = run_bm3d_step_compat(
        standard_layout(input_noisy.as_array()).view(),
        standard_layout(input_pilot.as_array()).view(),
        Bm3dMode::HardThreshold,
        standard_layout(sigma_psd.as_array()).view(),
        standard_layout(sigma_map.as_array()).view(),
        sigma_random,
        threshold,
        patch_size,
        step_size,
        search_window,
        max_matches,
        search_step,
        &plans,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(output.to_pyarray(py))
}

/// Wiener filtering step of BM3D for a single 2D image.
///
/// `search_step` is the candidate-search stride; `None` uses `step_size`.
#[pyfunction]
#[pyo3(signature = (input_noisy, input_pilot, sigma_psd, sigma_map, sigma_random, patch_size, step_size, search_window, max_matches, search_step=None))]
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
    search_step: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let plans = bm3d_core::pipeline::Bm3dPlans::new(patch_size, max_matches);
    let output = run_bm3d_step_compat(
        standard_layout(input_noisy.as_array()).view(),
        standard_layout(input_pilot.as_array()).view(),
        Bm3dMode::Wiener,
        standard_layout(sigma_psd.as_array()).view(),
        standard_layout(sigma_map.as_array()).view(),
        sigma_random,
        0.0,
        patch_size,
        step_size,
        search_window,
        max_matches,
        search_step,
        &plans,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(output.to_pyarray(py))
}

/// Hard thresholding step of BM3D for a 3D stack of images.
///
/// `search_step` is the candidate-search stride; `None` uses `step_size`.
#[pyfunction]
#[pyo3(signature = (input_noisy, input_pilot, sigma_psd, sigma_map, sigma_random, threshold, patch_size, step_size, search_window, max_matches, progress_callback=None, search_step=None))]
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
    progress_callback: Option<Py<PyAny>>,
    search_step: Option<usize>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let noisy = standard_layout(input_noisy.as_array()).into_owned();
    let pilot = standard_layout(input_pilot.as_array()).into_owned();
    let psd = standard_layout(sigma_psd.as_array()).into_owned();
    let smap = standard_layout(sigma_map.as_array()).into_owned();
    let plans = bm3d_core::pipeline::Bm3dPlans::new(patch_size, max_matches);
    let output = py.detach(|| {
        let progress_fn: Option<ProgressFn> = progress_callback.map(make_progress_fn);
        run_bm3d_step_stack_compat(
            noisy.view(),
            pilot.view(),
            Bm3dMode::HardThreshold,
            psd.view(),
            smap.view(),
            sigma_random,
            threshold,
            patch_size,
            step_size,
            search_window,
            max_matches,
            search_step,
            &plans,
            progress_fn.as_deref(),
        )
    });
    let output = map_stack_error(output)?;
    Ok(output.to_pyarray(py))
}

/// Wiener filtering step of BM3D for a 3D stack of images.
///
/// `search_step` is the candidate-search stride; `None` uses `step_size`.
#[pyfunction]
#[pyo3(signature = (input_noisy, input_pilot, sigma_psd, sigma_map, sigma_random, patch_size, step_size, search_window, max_matches, progress_callback=None, search_step=None))]
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
    progress_callback: Option<Py<PyAny>>,
    search_step: Option<usize>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let noisy = standard_layout(input_noisy.as_array()).into_owned();
    let pilot = standard_layout(input_pilot.as_array()).into_owned();
    let psd = standard_layout(sigma_psd.as_array()).into_owned();
    let smap = standard_layout(sigma_map.as_array()).into_owned();
    let plans = bm3d_core::pipeline::Bm3dPlans::new(patch_size, max_matches);
    let output = py.detach(|| {
        let progress_fn: Option<ProgressFn> = progress_callback.map(make_progress_fn);
        run_bm3d_step_stack_compat(
            noisy.view(),
            pilot.view(),
            Bm3dMode::Wiener,
            psd.view(),
            smap.view(),
            sigma_random,
            0.0,
            patch_size,
            step_size,
            search_window,
            max_matches,
            search_step,
            &plans,
            progress_fn.as_deref(),
        )
    });
    let output = map_stack_error(output)?;
    Ok(output.to_pyarray(py))
}

#[pyfunction]
pub fn test_block_matching_rust(
    input: PyReadonlyArray2<'_, f32>,
    ref_r: usize,
    ref_c: usize,
    patch_size: usize,
    search_win: usize,
    max_matches: usize,
) -> PyResult<Vec<(usize, usize, f32)>> {
    let (rows, cols) = input.as_array().dim();
    check_image_size(rows, cols, patch_size).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let result = bm3d_core::pipeline::test_block_matching(
        standard_layout(input.as_array()).view(),
        ref_r,
        ref_c,
        patch_size,
        search_win,
        max_matches,
    );
    Ok(result)
}

/// Estimate the static vertical streak profile.
#[pyfunction]
#[pyo3(name = "estimate_streak_profile_rust")]
pub fn estimate_streak_profile_py<'py>(
    py: Python<'py>,
    sinogram: PyReadonlyArray2<'py, f32>,
    sigma_smooth: f32,
    iterations: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let sinogram_view = standard_layout(sinogram.as_array());
    let result =
        bm3d_core::estimate_streak_profile_impl(sinogram_view.view(), sigma_smooth, iterations);
    Ok(result.to_pyarray(py))
}

/// Hard thresholding step of BM3D for a single 2D image (f64 precision).
///
/// `search_step` is the candidate-search stride; `None` uses `step_size`.
#[pyfunction]
#[pyo3(signature = (input_noisy, input_pilot, sigma_psd, sigma_map, sigma_random, threshold, patch_size, step_size, search_window, max_matches, search_step=None))]
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
    search_step: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let plans = bm3d_core::pipeline::Bm3dPlans::new(patch_size, max_matches);
    let output = run_bm3d_step_compat(
        standard_layout(input_noisy.as_array()).view(),
        standard_layout(input_pilot.as_array()).view(),
        Bm3dMode::HardThreshold,
        standard_layout(sigma_psd.as_array()).view(),
        standard_layout(sigma_map.as_array()).view(),
        sigma_random,
        threshold,
        patch_size,
        step_size,
        search_window,
        max_matches,
        search_step,
        &plans,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(output.to_pyarray(py))
}

/// Wiener filtering step of BM3D for a single 2D image (f64 precision).
///
/// `search_step` is the candidate-search stride; `None` uses `step_size`.
#[pyfunction]
#[pyo3(signature = (input_noisy, input_pilot, sigma_psd, sigma_map, sigma_random, patch_size, step_size, search_window, max_matches, search_step=None))]
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
    search_step: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let plans = bm3d_core::pipeline::Bm3dPlans::new(patch_size, max_matches);
    let output = run_bm3d_step_compat(
        standard_layout(input_noisy.as_array()).view(),
        standard_layout(input_pilot.as_array()).view(),
        Bm3dMode::Wiener,
        standard_layout(sigma_psd.as_array()).view(),
        standard_layout(sigma_map.as_array()).view(),
        sigma_random,
        0.0,
        patch_size,
        step_size,
        search_window,
        max_matches,
        search_step,
        &plans,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(output.to_pyarray(py))
}

/// Hard thresholding step of BM3D for a 3D stack of images (f64 precision).
///
/// `search_step` is the candidate-search stride; `None` uses `step_size`.
#[pyfunction]
#[pyo3(signature = (input_noisy, input_pilot, sigma_psd, sigma_map, sigma_random, threshold, patch_size, step_size, search_window, max_matches, progress_callback=None, search_step=None))]
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
    progress_callback: Option<Py<PyAny>>,
    search_step: Option<usize>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let noisy = standard_layout(input_noisy.as_array()).into_owned();
    let pilot = standard_layout(input_pilot.as_array()).into_owned();
    let psd = standard_layout(sigma_psd.as_array()).into_owned();
    let smap = standard_layout(sigma_map.as_array()).into_owned();
    let plans = bm3d_core::pipeline::Bm3dPlans::new(patch_size, max_matches);
    let output = py.detach(|| {
        let progress_fn: Option<ProgressFn> = progress_callback.map(make_progress_fn);
        run_bm3d_step_stack_compat(
            noisy.view(),
            pilot.view(),
            Bm3dMode::HardThreshold,
            psd.view(),
            smap.view(),
            sigma_random,
            threshold,
            patch_size,
            step_size,
            search_window,
            max_matches,
            search_step,
            &plans,
            progress_fn.as_deref(),
        )
    });
    let output = map_stack_error(output)?;
    Ok(output.to_pyarray(py))
}

/// Wiener filtering step of BM3D for a 3D stack of images (f64 precision).
///
/// `search_step` is the candidate-search stride; `None` uses `step_size`.
#[pyfunction]
#[pyo3(signature = (input_noisy, input_pilot, sigma_psd, sigma_map, sigma_random, patch_size, step_size, search_window, max_matches, progress_callback=None, search_step=None))]
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
    progress_callback: Option<Py<PyAny>>,
    search_step: Option<usize>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let noisy = standard_layout(input_noisy.as_array()).into_owned();
    let pilot = standard_layout(input_pilot.as_array()).into_owned();
    let psd = standard_layout(sigma_psd.as_array()).into_owned();
    let smap = standard_layout(sigma_map.as_array()).into_owned();
    let plans = bm3d_core::pipeline::Bm3dPlans::new(patch_size, max_matches);
    let output = py.detach(|| {
        let progress_fn: Option<ProgressFn> = progress_callback.map(make_progress_fn);
        run_bm3d_step_stack_compat(
            noisy.view(),
            pilot.view(),
            Bm3dMode::Wiener,
            psd.view(),
            smap.view(),
            sigma_random,
            0.0,
            patch_size,
            step_size,
            search_window,
            max_matches,
            search_step,
            &plans,
            progress_fn.as_deref(),
        )
    });
    let output = map_stack_error(output)?;
    Ok(output.to_pyarray(py))
}

#[pyfunction]
pub fn test_block_matching_rust_f64(
    input: PyReadonlyArray2<'_, f64>,
    ref_r: usize,
    ref_c: usize,
    patch_size: usize,
    search_win: usize,
    max_matches: usize,
) -> PyResult<Vec<(usize, usize, f64)>> {
    let (rows, cols) = input.as_array().dim();
    check_image_size(rows, cols, patch_size).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let result = bm3d_core::pipeline::test_block_matching(
        standard_layout(input.as_array()).view(),
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
    let sinogram_view = standard_layout(sinogram.as_array());
    let result =
        bm3d_core::estimate_streak_profile_impl(sinogram_view.view(), sigma_smooth, iterations);
    Ok(result.to_pyarray(py))
}

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
    notch_width = None,
    search_step = None
))]
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
    search_step: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let ring_mode = match mode.to_lowercase().as_str() {
        "generic" => RingRemovalMode::Generic,
        "streak" => RingRemovalMode::Streak,
        "multiscale_streak" | "multiscalestreak" => RingRemovalMode::MultiscaleStreak,
        "fourier_svd" | "fouriersvd" => RingRemovalMode::FourierSvd,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid mode '{}'. Expected 'generic', 'streak', 'multiscale_streak', or 'fourier_svd'.",
                mode
            )));
        }
    };

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
    // None keeps the per-mode default of Bm3dConfig.
    config.search_step = search_step;

    let output = bm3d_ring_artifact_removal(
        standard_layout(sinogram.as_array()).view(),
        ring_mode,
        &config,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok(output.to_pyarray(py))
}

/// Unified BM3D ring artifact removal for 2D sinograms (f64 precision).
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
    notch_width = None,
    search_step = None
))]
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
    search_step: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let ring_mode = match mode.to_lowercase().as_str() {
        "generic" => RingRemovalMode::Generic,
        "streak" => RingRemovalMode::Streak,
        "multiscale_streak" | "multiscalestreak" => RingRemovalMode::MultiscaleStreak,
        "fourier_svd" | "fouriersvd" => RingRemovalMode::FourierSvd,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid mode '{}'. Expected 'generic', 'streak', 'multiscale_streak', or 'fourier_svd'.",
                mode
            )));
        }
    };

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
    // None keeps the per-mode default of Bm3dConfig.
    config.search_step = search_step;

    let output = bm3d_ring_artifact_removal(
        standard_layout(sinogram.as_array()).view(),
        ring_mode,
        &config,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok(output.to_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "fourier_svd_removal_rust")]
#[pyo3(signature = (sinogram, fft_alpha=1.0, notch_width=2.0))]
pub fn fourier_svd_removal_py<'py>(
    py: Python<'py>,
    sinogram: PyReadonlyArray2<'py, f32>,
    fft_alpha: f32,
    notch_width: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let output = bm3d_core::fourier_svd::fourier_svd_removal(
        standard_layout(sinogram.as_array()).view(),
        fft_alpha,
        notch_width,
    );
    Ok(output.to_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "fourier_svd_removal_rust_f64")]
#[pyo3(signature = (sinogram, fft_alpha=1.0, notch_width=2.0))]
pub fn fourier_svd_removal_py_f64<'py>(
    py: Python<'py>,
    sinogram: PyReadonlyArray2<'py, f64>,
    fft_alpha: f64,
    notch_width: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let output = bm3d_core::fourier_svd::fourier_svd_removal(
        standard_layout(sinogram.as_array()).view(),
        fft_alpha,
        notch_width,
    );
    Ok(output.to_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "estimate_noise_sigma_rust")]
pub fn estimate_noise_sigma_py(sinogram: PyReadonlyArray2<'_, f32>) -> PyResult<f32> {
    Ok(bm3d_core::estimate_noise_sigma(
        standard_layout(sinogram.as_array()).view(),
    ))
}

/// Estimate noise standard deviation using MAD-based robust estimation (f64).
#[pyfunction]
#[pyo3(name = "estimate_noise_sigma_rust_f64")]
pub fn estimate_noise_sigma_py_f64(sinogram: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
    Ok(bm3d_core::estimate_noise_sigma(
        standard_layout(sinogram.as_array()).view(),
    ))
}

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
    psd_width = None,
    log_domain_input = None,
    search_step = None
))]
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
    log_domain_input: Option<bool>,
    search_step: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let default = MultiscaleConfig::<f32>::default();
    let mut config = MultiscaleConfig {
        num_scales,
        filter_strength: filter_strength.unwrap_or(default.filter_strength),
        threshold: threshold.unwrap_or(default.threshold),
        debin_iterations: debin_iterations.unwrap_or(default.debin_iterations),
        log_domain_input: log_domain_input.unwrap_or(default.log_domain_input),
        bm3d_config: default.bm3d_config,
    };

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
    // None keeps the per-mode default of Bm3dConfig.
    config.bm3d_config.search_step = search_step;

    let output =
        multiscale_bm3d_streak_removal(standard_layout(sinogram.as_array()).view(), &config)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok(output.to_pyarray(py))
}

/// Multi-scale BM3D streak removal for 2D sinograms (f64 precision).
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
    psd_width = None,
    log_domain_input = None,
    search_step = None
))]
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
    log_domain_input: Option<bool>,
    search_step: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let default = MultiscaleConfig::<f64>::default();
    let mut config = MultiscaleConfig {
        num_scales,
        filter_strength: filter_strength.unwrap_or(default.filter_strength),
        threshold: threshold.unwrap_or(default.threshold),
        debin_iterations: debin_iterations.unwrap_or(default.debin_iterations),
        log_domain_input: log_domain_input.unwrap_or(default.log_domain_input),
        bm3d_config: default.bm3d_config,
    };

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
    // None keeps the per-mode default of Bm3dConfig.
    config.bm3d_config.search_step = search_step;

    let output =
        multiscale_bm3d_streak_removal(standard_layout(sinogram.as_array()).view(), &config)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok(output.to_pyarray(py))
}

#[pymodule]
fn bm3d_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
