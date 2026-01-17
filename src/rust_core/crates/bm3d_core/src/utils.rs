use crate::float_trait::Bm3dFloat;
use ndarray::ArrayView1;

/// Compute 1D median filter with padding.
/// Mirrors `scipy.ndimage.median_filter` with mode='reflect' (or nearest).
/// For simplicity, we use replication padding or mirror.
pub fn compute_1d_median_filter<F: Bm3dFloat>(input: &[F], window_size: usize) -> Vec<F> {
    let n = input.len();
    let radius = window_size / 2;
    let mut output = Vec::with_capacity(n);
    let mut window_buffer = Vec::with_capacity(window_size);

    for i in 0..n {
        window_buffer.clear();
        for r in 0..window_size {
            // Replicate padding
            let idx = (i as isize + r as isize - radius as isize)
                .max(0)
                .min((n - 1) as isize) as usize;
            window_buffer.push(input[idx]);
        }

        // Find median
        // Sort using total_cmp (via to_f64)
        window_buffer.sort_by(|a, b| a.to_f64().unwrap().total_cmp(&b.to_f64().unwrap()));
        output.push(window_buffer[window_size / 2]);
    }
    output
}

/// Robust Sigma Estimation (MAD / 0.6745)
pub fn estimate_robust_sigma<F: Bm3dFloat>(data: ArrayView1<F>) -> f64 {
    // Collect into Vec to sort
    let mut vec_data: Vec<f64> = data.iter().map(|x| x.to_f64().unwrap()).collect();
    vec_data.sort_by(|a, b| a.total_cmp(b));

    let n = vec_data.len();
    if n == 0 {
        return 0.0;
    }

    let median = vec_data[n / 2];

    // Compute diffs
    let mut diffs: Vec<f64> = data
        .iter()
        .map(|x| (x.to_f64().unwrap() - median).abs())
        .collect();
    diffs.sort_by(|a, b| a.total_cmp(b));

    let mad = diffs[n / 2];

    if mad == 0.0 {
        // Fallback to std dev if MAD is zero
        // Manual std dev on f64 vec
        let mean = vec_data.iter().sum::<f64>() / n as f64;
        let variance =
            vec_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0); // Sample std dev
        variance.sqrt()
    } else {
        mad / 0.6745
    }
}
