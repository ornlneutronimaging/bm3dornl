//! PyO3 Python bindings for BM3D denoising.
//!
//! This crate provides thin Python bindings for the bm3d_core library.
//! All algorithm logic is in bm3d_core; this crate only handles
//! Python/NumPy type conversions.
//!
//! Both f32 and f64 precision are supported. Functions with `_f64` suffix
//! accept and return float64 numpy arrays.

use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray2, PyReadonlyArray3, PyArray3, ToPyArray, PyArray1};

use bm3d_core::{Bm3dMode, run_bm3d_step, run_bm3d_step_stack};

/// Hard thresholding step of BM3D for a single 2D image.
#[pyfunction]
pub fn bm3d_hard_thresholding<'py>(
    py: Python<'py>,
    input_noisy: PyReadonlyArray2<f32>,
    input_pilot: PyReadonlyArray2<f32>,
    sigma_psd: PyReadonlyArray2<f32>,
    sigma_map: PyReadonlyArray2<f32>,
    sigma_random: f32,
    threshold: f32,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> PyResult<&'py PyArray2<f32>> {
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
        max_matches
    ).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    Ok(output.to_pyarray(py))
}

/// Wiener filtering step of BM3D for a single 2D image.
#[pyfunction]
pub fn bm3d_wiener_filtering<'py>(
    py: Python<'py>,
    input_noisy: PyReadonlyArray2<f32>,
    input_pilot: PyReadonlyArray2<f32>,
    sigma_psd: PyReadonlyArray2<f32>,
    sigma_map: PyReadonlyArray2<f32>,
    sigma_random: f32,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> PyResult<&'py PyArray2<f32>> {
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
        max_matches
    ).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    Ok(output.to_pyarray(py))
}

/// Hard thresholding step of BM3D for a 3D stack of images.
#[pyfunction]
pub fn bm3d_hard_thresholding_stack<'py>(
    py: Python<'py>,
    input_noisy: PyReadonlyArray3<f32>,
    input_pilot: PyReadonlyArray3<f32>,
    sigma_psd: PyReadonlyArray2<f32>,
    sigma_map: PyReadonlyArray3<f32>,
    sigma_random: f32,
    threshold: f32,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> PyResult<&'py PyArray3<f32>> {
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
        max_matches
    ).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    Ok(output.to_pyarray(py))
}

/// Wiener filtering step of BM3D for a 3D stack of images.
#[pyfunction]
pub fn bm3d_wiener_filtering_stack<'py>(
    py: Python<'py>,
    input_noisy: PyReadonlyArray3<f32>,
    input_pilot: PyReadonlyArray3<f32>,
    sigma_psd: PyReadonlyArray2<f32>,
    sigma_map: PyReadonlyArray3<f32>,
    sigma_random: f32,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> PyResult<&'py PyArray3<f32>> {
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
        max_matches
    ).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    Ok(output.to_pyarray(py))
}

/// Test function for block matching (used for debugging/validation).
#[pyfunction]
pub fn test_block_matching_rust(
    input: PyReadonlyArray2<f32>,
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
pub fn estimate_streak_profile_py(
    py: Python<'_>,
    sinogram: PyReadonlyArray2<f32>,
    sigma_smooth: f32,
    iterations: usize,
) -> PyResult<Py<PyArray1<f32>>> {
    let sinogram_view = sinogram.as_array();
    let result = bm3d_core::estimate_streak_profile_impl(sinogram_view, sigma_smooth, iterations);
    Ok(PyArray1::from_owned_array(py, result).into())
}

// ============================================================================
// f64 (double precision) variants
// ============================================================================

/// Hard thresholding step of BM3D for a single 2D image (f64 precision).
#[pyfunction]
pub fn bm3d_hard_thresholding_f64<'py>(
    py: Python<'py>,
    input_noisy: PyReadonlyArray2<f64>,
    input_pilot: PyReadonlyArray2<f64>,
    sigma_psd: PyReadonlyArray2<f64>,
    sigma_map: PyReadonlyArray2<f64>,
    sigma_random: f64,
    threshold: f64,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> PyResult<&'py PyArray2<f64>> {
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
        max_matches
    ).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    Ok(output.to_pyarray(py))
}

/// Wiener filtering step of BM3D for a single 2D image (f64 precision).
#[pyfunction]
pub fn bm3d_wiener_filtering_f64<'py>(
    py: Python<'py>,
    input_noisy: PyReadonlyArray2<f64>,
    input_pilot: PyReadonlyArray2<f64>,
    sigma_psd: PyReadonlyArray2<f64>,
    sigma_map: PyReadonlyArray2<f64>,
    sigma_random: f64,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> PyResult<&'py PyArray2<f64>> {
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
        max_matches
    ).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    Ok(output.to_pyarray(py))
}

/// Hard thresholding step of BM3D for a 3D stack of images (f64 precision).
#[pyfunction]
pub fn bm3d_hard_thresholding_stack_f64<'py>(
    py: Python<'py>,
    input_noisy: PyReadonlyArray3<f64>,
    input_pilot: PyReadonlyArray3<f64>,
    sigma_psd: PyReadonlyArray2<f64>,
    sigma_map: PyReadonlyArray3<f64>,
    sigma_random: f64,
    threshold: f64,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> PyResult<&'py PyArray3<f64>> {
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
        max_matches
    ).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    Ok(output.to_pyarray(py))
}

/// Wiener filtering step of BM3D for a 3D stack of images (f64 precision).
#[pyfunction]
pub fn bm3d_wiener_filtering_stack_f64<'py>(
    py: Python<'py>,
    input_noisy: PyReadonlyArray3<f64>,
    input_pilot: PyReadonlyArray3<f64>,
    sigma_psd: PyReadonlyArray2<f64>,
    sigma_map: PyReadonlyArray3<f64>,
    sigma_random: f64,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> PyResult<&'py PyArray3<f64>> {
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
        max_matches
    ).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    Ok(output.to_pyarray(py))
}

/// Test function for block matching (f64 precision).
#[pyfunction]
pub fn test_block_matching_rust_f64(
    input: PyReadonlyArray2<f64>,
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
pub fn estimate_streak_profile_py_f64(
    py: Python<'_>,
    sinogram: PyReadonlyArray2<f64>,
    sigma_smooth: f64,
    iterations: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let sinogram_view = sinogram.as_array();
    let result = bm3d_core::estimate_streak_profile_impl(sinogram_view, sigma_smooth, iterations);
    Ok(PyArray1::from_owned_array(py, result).into())
}

/// BM3D Rust accelerator module
#[pymodule]
fn bm3d_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    // f32 (single precision) functions
    m.add_function(wrap_pyfunction!(bm3d_hard_thresholding, m)?)?;
    m.add_function(wrap_pyfunction!(bm3d_wiener_filtering, m)?)?;
    m.add_function(wrap_pyfunction!(bm3d_hard_thresholding_stack, m)?)?;
    m.add_function(wrap_pyfunction!(bm3d_wiener_filtering_stack, m)?)?;
    m.add_function(wrap_pyfunction!(test_block_matching_rust, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_streak_profile_py, m)?)?;

    // f64 (double precision) functions
    m.add_function(wrap_pyfunction!(bm3d_hard_thresholding_f64, m)?)?;
    m.add_function(wrap_pyfunction!(bm3d_wiener_filtering_f64, m)?)?;
    m.add_function(wrap_pyfunction!(bm3d_hard_thresholding_stack_f64, m)?)?;
    m.add_function(wrap_pyfunction!(bm3d_wiener_filtering_stack_f64, m)?)?;
    m.add_function(wrap_pyfunction!(test_block_matching_rust_f64, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_streak_profile_py_f64, m)?)?;
    Ok(())
}
