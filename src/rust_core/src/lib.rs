use pyo3::prelude::*;

mod block_matching;
mod filtering;
mod pipeline;
mod streak;
mod transforms;

/// BM3D Rust accelerator module
#[pymodule]
fn bm3d_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pipeline::bm3d_hard_thresholding, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline::bm3d_wiener_filtering, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline::bm3d_hard_thresholding_stack, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline::bm3d_wiener_filtering_stack, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline::test_block_matching_rust, m)?)?;
    m.add_function(wrap_pyfunction!(streak::estimate_streak_profile_py, m)?)?;
    Ok(())
}
