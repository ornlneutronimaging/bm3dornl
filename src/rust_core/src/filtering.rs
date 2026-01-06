use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use ndarray::Zip;

/// Hard Thresholding via shrinkage.
/// Zeros out coefficients with magnitude <= threshold.
pub fn hard_thresholding(
    input: &mut Array3<f32>,
    threshold: f32,
) {
    input.map_inplace(|v| {
        if v.abs() <= threshold {
            *v = 0.0;
        }
    });
}

/// Collaborative Wiener Filtering.
/// 
/// # Arguments
/// * `noisy_group` - The group of noisy patches (transformed).
/// * `pilot_group` - The group of pilot patches (transformed) from previous step.
/// * `noise_sigma` - Standard deviation of noise.
/// 
/// Returns simple element-wise Wiener multiplication.
pub fn wiener_filtering(
    noisy_group: &Array3<f32>,
    pilot_group: &Array3<f32>,
    noise_sigma: f32,
) -> Array3<f32> {
    let mut filtered = Array3::<f32>::zeros(noisy_group.dim());
    let sigma_sq = noise_sigma * noise_sigma;
    
    // Vectorized zip
    Zip::from(&mut filtered)
        .and(noisy_group)
        .and(pilot_group)
        .for_each(|f, &n, &p| {
            let p_sq = p * p;
            let factor = p_sq / (p_sq + sigma_sq + 1e-8);
            *f = n * factor;
        });
        
    filtered
}

