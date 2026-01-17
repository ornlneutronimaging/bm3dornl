use crate::float_trait::Bm3dFloat;
use ndarray::{Array2, ArrayView2};

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

/// Estimate noise standard deviation using MAD-based robust estimation.
///
/// This implements the sigma estimation from Mäkinen et al. (2021).
/// The image is filtered to isolate vertical streaks (vertical Gaussian + horizontal High-pass),
/// then the MAD (Median Absolute Deviation) is computed and scaled.
pub fn estimate_noise_sigma<F: Bm3dFloat>(sinogram: ArrayView2<F>) -> F {
    let (rows, _cols) = sinogram.dim();

    // Step 1: Vertical Gaussian filter
    // Sigma = height / 12.0
    let sigma_v = rows as f64 / 12.0;

    // Create Gaussian kernel
    // Support width: usually +/- 4 sigma is enough for f32/f64
    let radius = (4.0 * sigma_v).ceil() as usize;
    let width = 2 * radius + 1;
    let mut kernel = Vec::with_capacity(width);
    let mut sum = 0.0;

    for i in 0..width {
        let x = i as f64 - radius as f64;
        let val = (-0.5 * (x / sigma_v).powi(2)).exp();
        kernel.push(F::from_f64_c(val));
        sum += val;
    }

    // Normalize kernel
    let sum_f = F::from_f64_c(sum);
    for k in &mut kernel {
        *k /= sum_f;
    }

    // Apply Vertical Convolution
    let smoothed = gaussian_filter_1d_vertical(sinogram, &kernel);

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

fn gaussian_filter_1d_vertical<F: Bm3dFloat>(data: ArrayView2<F>, kernel: &[F]) -> Array2<F> {
    let (rows, cols) = data.dim();
    let k_len = kernel.len();
    let radius = k_len / 2;
    let mut output = Array2::zeros((rows, cols));

    // For large images, this could be parallelized, but keep simple for now
    for c in 0..cols {
        for r in 0..rows {
            let mut sum = F::zero();
            for (k, &k_val) in kernel.iter().enumerate() {
                let k_idx = k as isize - radius as isize;
                let src_r = (r as isize + k_idx).clamp(0, (rows - 1) as isize);
                sum += data[[src_r as usize, c]] * k_val;
            }
            output[[r, c]] = sum;
        }
    }
    output
}

fn convolve_1d_horizontal<F: Bm3dFloat>(data: ArrayView2<F>, kernel: &[F]) -> Array2<F> {
    let (rows, cols) = data.dim();
    let k_len = kernel.len();
    let radius = k_len / 2;
    let mut output = Array2::zeros((rows, cols));

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
    // Flatten array
    let mut flat_data: Vec<F> = data.iter().cloned().collect();
    let median = median_of_slice(&mut flat_data);

    // Compute absolute deviations
    let mut deviations: Vec<F> = flat_data.iter().map(|&x| (x - median).abs()).collect();

    median_of_slice(&mut deviations)
}

#[cfg(test)]
#[allow(clippy::print_stdout)]
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
