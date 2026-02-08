use crate::float_trait::Bm3dFloat;
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;
use std::env;

/// Daubechies-3 wavelet high-pass decomposition filter coefficients.
/// These coefficients correspond to the decomposition high-pass filter (dec_hi).
/// L2 norm is 1.0.
const DB3_DEC_HI: [f64; 6] = [
    -0.33267055,
    0.80689151,
    -0.45987750,
    -0.13501102,
    0.08544127,
    0.03522629,
];

/// Default cap for columns used in automatic sigma estimation.
///
/// For very wide sinograms, sampling evenly spaced columns preserves
/// robust scale estimation while reducing runtime significantly.
const DEFAULT_SIGMA_EST_MAX_COLUMNS: usize = 1024;
const SIGMA_EST_MAX_COLUMNS_ENV: &str = "BM3D_SIGMA_EST_MAX_COLUMNS";

fn resolve_sigma_est_max_columns() -> Option<usize> {
    match env::var(SIGMA_EST_MAX_COLUMNS_ENV)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        Some(0) => None,
        Some(v) => Some(v),
        None => Some(DEFAULT_SIGMA_EST_MAX_COLUMNS),
    }
}

fn sample_columns_evenly<F: Bm3dFloat>(data: ArrayView2<F>, sample_cols: usize) -> Array2<F> {
    let (rows, cols) = data.dim();
    let sample_cols = sample_cols.max(1).min(cols);
    if sample_cols == cols {
        return data.to_owned();
    }

    Array2::from_shape_fn((rows, sample_cols), |(r, i)| {
        let c = if sample_cols == 1 {
            cols / 2
        } else {
            i * (cols - 1) / (sample_cols - 1)
        };
        data[[r, c]]
    })
}

/// Estimate noise standard deviation using MAD-based robust estimation.
///
/// This implements the sigma estimation from Mäkinen et al. (2021).
/// The image is filtered to isolate vertical streaks (vertical Gaussian + horizontal High-pass),
/// then the MAD (Median Absolute Deviation) is computed and scaled.
pub fn estimate_noise_sigma<F: Bm3dFloat>(sinogram: ArrayView2<F>) -> F {
    let (_rows, cols) = sinogram.dim();
    let sampled_storage = match resolve_sigma_est_max_columns() {
        Some(max_cols) if cols > max_cols => Some(sample_columns_evenly(sinogram, max_cols)),
        _ => None,
    };
    let (rows, _cols) = if let Some(sampled) = sampled_storage.as_ref() {
        sampled.dim()
    } else {
        sinogram.dim()
    };

    // Step 1: Vertical Gaussian filter
    // Sigma = height / 12.0
    //
    // Use a recursive IIR approximation (Young/van Vliet family coefficients)
    // instead of wide FIR convolution. This preserves per-sinogram semantics
    // while reducing complexity from O(rows*cols*kernel_width) to O(rows*cols).
    let sigma_v = rows as f64 / 12.0;
    let smoothed = if let Some(sampled) = sampled_storage.as_ref() {
        gaussian_filter_1d_vertical_recursive(sampled.view(), sigma_v)
    } else {
        gaussian_filter_1d_vertical_recursive(sinogram, sigma_v)
    };

    // Step 2: Horizontal High-Pass (db3)
    // Convert DB3 coeffs to generic float
    let db3_kernel: Vec<F> = DB3_DEC_HI.iter().map(|&x| F::from_f64_c(x)).collect();
    let filtered = convolve_1d_horizontal(smoothed.view(), &db3_kernel);

    // Step 3: Compute MAD
    let mad_val = compute_mad(filtered.view());

    // Step 4: Scale to Sigma
    // Filter gain for db3 high-pass is L2 norm = 1.0.
    // Vertical filter gain for vertical streaks (constant signal) is L1 norm = 1.0.
    // So Effective Gain = 1.0.
    // sigma = 1.4826 * MAD

    mad_val * F::from_f64_c(1.4826)
}

#[inline]
fn recursive_gaussian_coefficients(sigma: f64) -> (f64, f64, f64, f64, f64) {
    // Coefficients from a widely used recursive Gaussian approximation.
    // q maps sigma to a stable coefficient domain.
    let sigma = sigma.max(0.5);
    let q = if sigma >= 2.5 {
        0.98711 * sigma - 0.96330
    } else {
        3.97156 - 4.14554 * (1.0 - 0.26891 * sigma).sqrt()
    };

    let b0 = 1.57825 + 2.44413 * q + 1.4281 * q * q + 0.422205 * q * q * q;
    let b1 = 2.44413 * q + 2.85619 * q * q + 1.26661 * q * q * q;
    let b2 = -(1.4281 * q * q + 1.26661 * q * q * q);
    let b3 = 0.422205 * q * q * q;
    let gain = 1.0 - (b1 + b2 + b3) / b0;
    (b0, b1, b2, b3, gain)
}

fn gaussian_filter_1d_vertical_recursive<F: Bm3dFloat>(
    data: ArrayView2<F>,
    sigma_v: f64,
) -> Array2<F> {
    let (rows, cols) = data.dim();
    if rows == 0 || cols == 0 {
        return Array2::zeros((rows, cols));
    }

    let (b0, b1, b2, b3, gain) = recursive_gaussian_coefficients(sigma_v);
    let b0f = F::from_f64_c(b0);
    let b1f = F::from_f64_c(b1);
    let b2f = F::from_f64_c(b2);
    let b3f = F::from_f64_c(b3);
    let gain_f = F::from_f64_c(gain);

    let mut output = Array2::zeros((rows, cols));
    let mut forward = vec![F::zero(); rows];

    if let (Some(data_slice), Some(output_slice)) = (
        data.as_slice_memory_order(),
        output.as_slice_memory_order_mut(),
    ) {
        for c in 0..cols {
            let x0 = data_slice[c];
            let mut f1 = x0;
            let mut f2 = x0;
            let mut f3 = x0;

            for r in 0..rows {
                let x = data_slice[r * cols + c];
                let rec = (b1f * f1 + b2f * f2 + b3f * f3) / b0f;
                let f0 = gain_f * x + rec;
                forward[r] = f0;
                f3 = f2;
                f2 = f1;
                f1 = f0;
            }

            let mut b1s = forward[rows - 1];
            let mut b2s = b1s;
            let mut b3s = b1s;
            for r in (0..rows).rev() {
                let x = forward[r];
                let rec = (b1f * b1s + b2f * b2s + b3f * b3s) / b0f;
                let y = gain_f * x + rec;
                output_slice[r * cols + c] = y;
                b3s = b2s;
                b2s = b1s;
                b1s = y;
            }
        }
    } else {
        for c in 0..cols {
            let x0 = data[[0, c]];
            let mut f1 = x0;
            let mut f2 = x0;
            let mut f3 = x0;

            for r in 0..rows {
                let x = data[[r, c]];
                let rec = (b1f * f1 + b2f * f2 + b3f * f3) / b0f;
                let f0 = gain_f * x + rec;
                forward[r] = f0;
                f3 = f2;
                f2 = f1;
                f1 = f0;
            }

            let mut b1s = forward[rows - 1];
            let mut b2s = b1s;
            let mut b3s = b1s;
            for r in (0..rows).rev() {
                let x = forward[r];
                let rec = (b1f * b1s + b2f * b2s + b3f * b3s) / b0f;
                let y = gain_f * x + rec;
                output[[r, c]] = y;
                b3s = b2s;
                b2s = b1s;
                b1s = y;
            }
        }
    }

    output
}

fn convolve_1d_horizontal<F: Bm3dFloat>(data: ArrayView2<F>, kernel: &[F]) -> Array2<F> {
    let (rows, cols) = data.dim();
    let k_len = kernel.len();
    let radius = k_len / 2;
    let mut output = Array2::zeros((rows, cols));

    if let (Some(data_slice), Some(output_slice)) = (
        data.as_slice_memory_order(),
        output.as_slice_memory_order_mut(),
    ) {
        output_slice
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(r, out_row)| {
                let row_base = r * cols;
                for (c, out_cell) in out_row.iter_mut().enumerate() {
                    let mut sum = F::zero();
                    for (k, &k_val) in kernel.iter().enumerate() {
                        let k_idx = k as isize - radius as isize;
                        let src_c = (c as isize + k_idx).clamp(0, (cols - 1) as isize) as usize;
                        sum += data_slice[row_base + src_c] * k_val;
                    }
                    *out_cell = sum;
                }
            });
    } else {
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
    }

    output
}

fn median_of_slice<F: Bm3dFloat>(data: &mut [F]) -> F {
    let len = data.len();
    if len == 0 {
        return F::zero();
    }
    let mid = len / 2;

    // select_nth_unstable finds the median in O(n)
    let (_, &mut median, _) = data.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());

    if len % 2 == 1 {
        median
    } else {
        // Even number of elements: (mid + mid-1) / 2
        // We need to find the element before mid.
        // select_nth_unstable partitioned the array so everything before mid is <= mid.
        let left_part = &data[..mid];
        let prev_median = left_part
            .iter()
            .fold(F::neg_infinity(), |a, &b| if b > a { b } else { a });
        (prev_median + median) / F::from_f64_c(2.0)
    }
}

fn compute_mad<F: Bm3dFloat>(data: ArrayView2<F>) -> F {
    // Flatten once and reuse the same buffer for absolute deviations to reduce
    // transient memory and allocation overhead in the hot auto-sigma path.
    let mut values: Vec<F> = data.iter().copied().collect();
    let median = median_of_slice(&mut values);
    for v in &mut values {
        *v = (*v - median).abs();
    }
    median_of_slice(&mut values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rand::prelude::*;
    use rand_distr::{Distribution, Normal};

    fn generate_vertical_streaks(
        height: usize,
        width: usize,
        sigma: f32,
        seed: u64,
    ) -> Array2<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, sigma).unwrap();

        // Generate 1D profile
        let mut profile = Vec::with_capacity(width);
        for _ in 0..width {
            profile.push(normal.sample(&mut rng));
        }

        // Broadcast to 2D
        let mut img = Array2::zeros((height, width));
        for r in 0..height {
            for c in 0..width {
                img[[r, c]] = profile[c];
            }
        }
        img
    }

    #[test]
    fn test_estimate_known_sigma_vertical_streaks() {
        let (height, width) = (256, 4096);
        let sigma_true = 0.05;
        let noise_img = generate_vertical_streaks(height, width, sigma_true, 42);

        let sigma_est = estimate_noise_sigma(noise_img.view());

        let error = (sigma_est - sigma_true).abs() / sigma_true;

        // Expect < 10% error
        assert!(
            error < 0.10,
            "Error {:.2}% exceeds 10% tolerance",
            error * 100.0
        );
    }

    #[test]
    fn test_relative_noise_levels() {
        let (height, width) = (128, 128);

        let sigma_low = 0.01;
        let img_low = generate_vertical_streaks(height, width, sigma_low, 42);

        let sigma_high = 0.10;
        let img_high = generate_vertical_streaks(height, width, sigma_high, 42);

        let est_low = estimate_noise_sigma(img_low.view());
        let est_high = estimate_noise_sigma(img_high.view());

        assert!(
            est_high > est_low,
            "High noise should yield higher sigma than low noise"
        );

        // Safe division check
        if est_low > 1e-9 {
            let ratio = est_high / est_low;
            assert!(
                ratio > 8.0 && ratio < 12.0,
                "Expected ratio ~10, got {}",
                ratio
            );
        }
    }

    #[test]
    fn test_horizontal_structure_insensitivity() {
        let (height, width) = (256, 256);

        // Strong horizontal structure: Y-varying signal (Sine wave down columns)
        // Image(r, c) = sin(r * scale)
        let mut img = Array2::<f32>::zeros((height, width));
        let rows = height as f32;

        for r in 0..height {
            let val = (r as f32 / rows * 10.0 * std::f32::consts::PI).sin(); // Amplitude 1.0
            for c in 0..width {
                img[[r, c]] = val;
            }
        }

        // Add weak vertical streaks
        let sigma_true = 0.02;
        let streaks = generate_vertical_streaks(height, width, sigma_true, 42);

        let combined = &img + &streaks;

        let sigma_est = estimate_noise_sigma(combined.view());

        // It might overestimate slightly, but shouldn't be dominated by structure (1.0)
        assert!(
            sigma_est < 0.10,
            "Estimator heavily affected by horizontal structure. Got {}",
            sigma_est
        );
    }

    #[test]
    fn test_f64_support() {
        // Verify f64 works
        // Use larger size for stable estimation (same as f32 test)
        let (height, width) = (256, 1024);
        let sigma_true = 0.05f64;

        // Generate f64 data
        let mut rng = StdRng::seed_from_u64(999);
        let normal = Normal::new(0.0, sigma_true).unwrap();
        let mut img = Array2::<f64>::zeros((height, width));

        for c in 0..width {
            let val = normal.sample(&mut rng);
            for r in 0..height {
                img[[r, c]] = val;
            }
        }

        let sigma_est = estimate_noise_sigma(img.view());
        let error = (sigma_est - sigma_true).abs() / sigma_true;

        assert!(
            error < 0.10,
            "f64 estimation failed with error {:.2}%",
            error * 100.0
        );
    }
}
