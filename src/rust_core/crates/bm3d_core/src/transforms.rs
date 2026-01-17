use ndarray::{Array2, ArrayView2};
use rustfft::{num_complex::Complex, Fft};
use std::sync::Arc;

use crate::float_trait::Bm3dFloat;

// =============================================================================
// Constants for Transform Operations
// =============================================================================

// Note: WHT normalization scale is now computed at runtime using F::usize_as(64)
// to support both f32 and f64 precision.

/// Compute 2D FFT of a square patch using pre-computed plans.
/// Returns unnormalized FFT.
pub fn fft2d<F: Bm3dFloat>(
    input: ArrayView2<F>,
    fft_row_plan: &Arc<dyn Fft<F>>,
    fft_col_plan: &Arc<dyn Fft<F>>,
) -> Array2<Complex<F>> {
    let (rows, cols) = input.dim();

    // 1. Transform rows
    let mut intermediate = Array2::<Complex<F>>::zeros((rows, cols));
    let mut row_vec = vec![Complex::new(F::zero(), F::zero()); cols];

    for r in 0..rows {
        // Copy to buffer
        for (c, &v) in input.row(r).iter().enumerate() {
            row_vec[c] = Complex::new(v, F::zero());
        }
        // FFT
        fft_row_plan.process(&mut row_vec);
        // Copy back
        for c in 0..cols {
            intermediate[[r, c]] = row_vec[c];
        }
    }

    // 2. Transform columns
    let mut output = Array2::<Complex<F>>::zeros((rows, cols));
    let mut col_vec = vec![Complex::new(F::zero(), F::zero()); rows];

    for c in 0..cols {
        // Extract column
        for r in 0..rows {
            col_vec[r] = intermediate[[r, c]];
        }
        // FFT
        fft_col_plan.process(&mut col_vec);
        // Copy back
        for r in 0..rows {
            output[[r, c]] = col_vec[r];
        }
    }

    output
}

/// Compute 2D Inverse FFT of a square patch using pre-computed plans.
/// Normalizes by 1/(rows*cols).
pub fn ifft2d<F: Bm3dFloat>(
    input: &Array2<Complex<F>>,
    ifft_row_plan: &Arc<dyn Fft<F>>,
    ifft_col_plan: &Arc<dyn Fft<F>>,
) -> Array2<F> {
    let (rows, cols) = input.dim();

    // 1. Transform columns
    let mut intermediate = input.clone();
    let mut col_vec = vec![Complex::new(F::zero(), F::zero()); rows];

    for c in 0..cols {
        for r in 0..rows {
            col_vec[r] = intermediate[[r, c]];
        }
        ifft_col_plan.process(&mut col_vec);
        for r in 0..rows {
            intermediate[[r, c]] = col_vec[r];
        }
    }

    // 2. Transform rows
    let mut output = Array2::<F>::zeros((rows, cols));
    let norm_factor = F::one() / F::usize_as(rows * cols);
    let mut row_vec = vec![Complex::new(F::zero(), F::zero()); cols];

    for r in 0..rows {
        for c in 0..cols {
            row_vec[c] = intermediate[[r, c]];
        }
        ifft_row_plan.process(&mut row_vec);
        for c in 0..cols {
            output[[r, c]] = row_vec[c].re * norm_factor;
        }
    }

    output
}

/// In-place Fast Walsh-Hadamard Transform (Natural Order) for 8 elements.
/// Uses a butterfly network with only additions and subtractions.
/// Complexity: 8 log2(8) = 24 ops.
/// This allows "multiplication-free" transform, drastically speeding up processing for 8x8 blocks.
#[inline(always)]
fn fwht8<F: Bm3dFloat>(buf: &mut [F; 8]) {
    // Stage 1 (Stride 1)
    let t0 = buf[0] + buf[1];
    buf[1] = buf[0] - buf[1];
    buf[0] = t0;
    let t2 = buf[2] + buf[3];
    buf[3] = buf[2] - buf[3];
    buf[2] = t2;
    let t4 = buf[4] + buf[5];
    buf[5] = buf[4] - buf[5];
    buf[4] = t4;
    let t6 = buf[6] + buf[7];
    buf[7] = buf[6] - buf[7];
    buf[6] = t6;

    // Stage 2 (Stride 2)
    let t0 = buf[0] + buf[2];
    buf[2] = buf[0] - buf[2];
    buf[0] = t0;
    let t1 = buf[1] + buf[3];
    buf[3] = buf[1] - buf[3];
    buf[1] = t1;
    let t4 = buf[4] + buf[6];
    buf[6] = buf[4] - buf[6];
    buf[4] = t4;
    let t5 = buf[5] + buf[7];
    buf[7] = buf[5] - buf[7];
    buf[5] = t5;

    // Stage 3 (Stride 4)
    let t0 = buf[0] + buf[4];
    buf[4] = buf[0] - buf[4];
    buf[0] = t0;
    let t1 = buf[1] + buf[5];
    buf[5] = buf[1] - buf[5];
    buf[1] = t1;
    let t2 = buf[2] + buf[6];
    buf[6] = buf[2] - buf[6];
    buf[2] = t2;
    let t3 = buf[3] + buf[7];
    buf[7] = buf[3] - buf[7];
    buf[3] = t3;
}

/// 2D WHT for 8x8 patch. Returns Complex (im=0) for compatibility.
pub fn wht2d_8x8_forward<F: Bm3dFloat>(input: ArrayView2<F>) -> Array2<Complex<F>> {
    let mut data = [F::zero(); 64];
    let mut idx = 0;
    for r in 0..8 {
        for c in 0..8 {
            data[idx] = input[[r, c]];
            idx += 1;
        }
    }
    for r in 0..8 {
        let mut row_buf = [F::zero(); 8];
        let offset = r * 8;
        row_buf.copy_from_slice(&data[offset..offset + 8]);
        fwht8(&mut row_buf);
        data[offset..offset + 8].copy_from_slice(&row_buf);
    }
    for c in 0..8 {
        let mut col_buf = [F::zero(); 8];
        for r in 0..8 {
            col_buf[r] = data[r * 8 + c];
        }
        fwht8(&mut col_buf);
        for r in 0..8 {
            data[r * 8 + c] = col_buf[r];
        }
    }
    let mut output = Array2::<Complex<F>>::zeros((8, 8));
    idx = 0;
    for r in 0..8 {
        for c in 0..8 {
            output[[r, c]] = Complex::new(data[idx], F::zero());
            idx += 1;
        }
    }
    output
}

/// 2D Inverse WHT for 8x8 patch.
pub fn wht2d_8x8_inverse<F: Bm3dFloat>(input: &Array2<Complex<F>>) -> Array2<F> {
    let mut data = [F::zero(); 64];
    let mut idx = 0;
    for r in 0..8 {
        for c in 0..8 {
            data[idx] = input[[r, c]].re;
            idx += 1;
        }
    }
    for r in 0..8 {
        let mut row_buf = [F::zero(); 8];
        let offset = r * 8;
        row_buf.copy_from_slice(&data[offset..offset + 8]);
        fwht8(&mut row_buf);
        data[offset..offset + 8].copy_from_slice(&row_buf);
    }
    for c in 0..8 {
        let mut col_buf = [F::zero(); 8];
        for r in 0..8 {
            col_buf[r] = data[r * 8 + c];
        }
        fwht8(&mut col_buf);
        for r in 0..8 {
            data[r * 8 + c] = col_buf[r];
        }
    }
    let norm_scale = F::one() / F::usize_as(64);
    let mut output = Array2::<F>::zeros((8, 8));
    idx = 0;
    for r in 0..8 {
        for c in 0..8 {
            output[[r, c]] = data[idx] * norm_scale;
            idx += 1;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rustfft::FftPlanner;

    // Helper: Simple Linear Congruential Generator for deterministic "random" test data
    // This avoids adding rand as a dependency while still providing varied test inputs
    struct SimpleLcg {
        state: u64,
    }

    impl SimpleLcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            // LCG parameters from Numerical Recipes
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.state
        }

        fn next_f32(&mut self) -> f32 {
            // Generate f32 in range [-1.0, 1.0)
            let u = self.next_u64();
            ((u >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }

        fn next_f64(&mut self) -> f64 {
            // Generate f64 in range [-1.0, 1.0)
            let u = self.next_u64();
            ((u >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    // Helper: Create FFT plans for a given size
    #[allow(clippy::type_complexity)]
    fn create_fft_plans_f32(
        rows: usize,
        cols: usize,
    ) -> (
        std::sync::Arc<dyn Fft<f32>>,
        std::sync::Arc<dyn Fft<f32>>,
        std::sync::Arc<dyn Fft<f32>>,
        std::sync::Arc<dyn Fft<f32>>,
    ) {
        let mut planner = FftPlanner::<f32>::new();
        let fft_row = planner.plan_fft_forward(cols);
        let fft_col = planner.plan_fft_forward(rows);
        let ifft_row = planner.plan_fft_inverse(cols);
        let ifft_col = planner.plan_fft_inverse(rows);
        (fft_row, fft_col, ifft_row, ifft_col)
    }

    #[allow(clippy::type_complexity)]
    fn create_fft_plans_f64(
        rows: usize,
        cols: usize,
    ) -> (
        std::sync::Arc<dyn Fft<f64>>,
        std::sync::Arc<dyn Fft<f64>>,
        std::sync::Arc<dyn Fft<f64>>,
        std::sync::Arc<dyn Fft<f64>>,
    ) {
        let mut planner = FftPlanner::<f64>::new();
        let fft_row = planner.plan_fft_forward(cols);
        let fft_col = planner.plan_fft_forward(rows);
        let ifft_row = planner.plan_fft_inverse(cols);
        let ifft_col = planner.plan_fft_inverse(rows);
        (fft_row, fft_col, ifft_row, ifft_col)
    }

    // Helper: Check if two f32 arrays are approximately equal
    fn arrays_approx_equal_f32(a: &Array2<f32>, b: &Array2<f32>, epsilon: f32) -> bool {
        if a.dim() != b.dim() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < epsilon)
    }

    fn arrays_approx_equal_f64(a: &Array2<f64>, b: &Array2<f64>, epsilon: f64) -> bool {
        if a.dim() != b.dim() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < epsilon)
    }

    // Helper: Generate deterministic "random" matrix
    fn random_matrix_f32(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
        let mut rng = SimpleLcg::new(seed);
        Array2::from_shape_fn((rows, cols), |_| rng.next_f32())
    }

    fn random_matrix_f64(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
        let mut rng = SimpleLcg::new(seed);
        Array2::from_shape_fn((rows, cols), |_| rng.next_f64())
    }

    // ==================== FFT Round-Trip Tests (f32) ====================

    #[test]
    fn test_fft2d_roundtrip_8x8() {
        let input = random_matrix_f32(8, 8, 12345);
        let (fft_row, fft_col, ifft_row, ifft_col) = create_fft_plans_f32(8, 8);

        let freq = fft2d(input.view(), &fft_row, &fft_col);
        let output = ifft2d(&freq, &ifft_row, &ifft_col);

        assert!(
            arrays_approx_equal_f32(&input, &output, 1e-5),
            "FFT roundtrip failed: max diff = {}",
            input
                .iter()
                .zip(output.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        );
    }

    #[test]
    fn test_fft2d_roundtrip_various_sizes() {
        let sizes = [(4, 4), (8, 8), (16, 16), (32, 32), (4, 8), (8, 16)];

        for (rows, cols) in sizes {
            let input = random_matrix_f32(rows, cols, (rows * 1000 + cols) as u64);
            let (fft_row, fft_col, ifft_row, ifft_col) = create_fft_plans_f32(rows, cols);

            let freq = fft2d(input.view(), &fft_row, &fft_col);
            let output = ifft2d(&freq, &ifft_row, &ifft_col);

            let max_diff = input
                .iter()
                .zip(output.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(
                arrays_approx_equal_f32(&input, &output, 1e-5),
                "FFT roundtrip failed for {}x{}: max diff = {}",
                rows,
                cols,
                max_diff
            );
        }
    }

    #[test]
    fn test_fft2d_roundtrip_multiple_seeds() {
        // Test with 10 different random inputs to increase confidence
        for seed in 0..10u64 {
            let input = random_matrix_f32(8, 8, seed * 7919); // Use prime multiplier for varied seeds
            let (fft_row, fft_col, ifft_row, ifft_col) = create_fft_plans_f32(8, 8);

            let freq = fft2d(input.view(), &fft_row, &fft_col);
            let output = ifft2d(&freq, &ifft_row, &ifft_col);

            assert!(
                arrays_approx_equal_f32(&input, &output, 1e-5),
                "FFT roundtrip failed for seed {}",
                seed
            );
        }
    }

    // ==================== FFT f64 Round-Trip Tests ====================

    #[test]
    fn test_fft2d_roundtrip_8x8_f64() {
        let input = random_matrix_f64(8, 8, 12345);
        let (fft_row, fft_col, ifft_row, ifft_col) = create_fft_plans_f64(8, 8);

        let freq = fft2d(input.view(), &fft_row, &fft_col);
        let output = ifft2d(&freq, &ifft_row, &ifft_col);

        assert!(
            arrays_approx_equal_f64(&input, &output, 1e-12),
            "FFT f64 roundtrip failed: max diff = {}",
            input
                .iter()
                .zip(output.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max)
        );
    }

    #[test]
    fn test_fft2d_roundtrip_various_sizes_f64() {
        let sizes = [(4, 4), (8, 8), (16, 16), (32, 32)];

        for (rows, cols) in sizes {
            let input = random_matrix_f64(rows, cols, (rows * 1000 + cols) as u64);
            let (fft_row, fft_col, ifft_row, ifft_col) = create_fft_plans_f64(rows, cols);

            let freq = fft2d(input.view(), &fft_row, &fft_col);
            let output = ifft2d(&freq, &ifft_row, &ifft_col);

            let max_diff = input
                .iter()
                .zip(output.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);

            assert!(
                arrays_approx_equal_f64(&input, &output, 1e-12),
                "FFT f64 roundtrip failed for {}x{}: max diff = {}",
                rows,
                cols,
                max_diff
            );
        }
    }

    // ==================== FFT Known-Value Tests ====================

    #[test]
    fn test_fft2d_zeros() {
        let input = Array2::<f32>::zeros((8, 8));
        let (fft_row, fft_col, _, _) = create_fft_plans_f32(8, 8);

        let output = fft2d(input.view(), &fft_row, &fft_col);

        // All frequency components should be zero
        for val in output.iter() {
            assert!(
                val.norm() < 1e-10,
                "FFT of zeros should be zeros, got magnitude {}",
                val.norm()
            );
        }
    }

    #[test]
    fn test_fft2d_constant() {
        // All ones: DC component should be N*M, others should be ~0
        let input = Array2::<f32>::ones((8, 8));
        let (fft_row, fft_col, _, _) = create_fft_plans_f32(8, 8);

        let output = fft2d(input.view(), &fft_row, &fft_col);

        // DC component (0,0) should equal sum of all inputs = 64
        let dc = output[[0, 0]];
        assert!(
            (dc.re - 64.0).abs() < 1e-5 && dc.im.abs() < 1e-5,
            "DC component should be 64+0i, got {:?}",
            dc
        );

        // All other components should be ~0
        for r in 0..8 {
            for c in 0..8 {
                if r != 0 || c != 0 {
                    let val = output[[r, c]];
                    assert!(
                        val.norm() < 1e-5,
                        "Non-DC component [{},{}] should be ~0, got magnitude {}",
                        r,
                        c,
                        val.norm()
                    );
                }
            }
        }
    }

    #[test]
    fn test_fft2d_impulse() {
        // Single 1.0 at (0,0), rest zeros
        // FFT of impulse should have constant magnitude across all bins
        let mut input = Array2::<f32>::zeros((8, 8));
        input[[0, 0]] = 1.0;
        let (fft_row, fft_col, _, _) = create_fft_plans_f32(8, 8);

        let output = fft2d(input.view(), &fft_row, &fft_col);

        // All frequency bins should have magnitude 1 (unnormalized FFT)
        for r in 0..8 {
            for c in 0..8 {
                let mag = output[[r, c]].norm();
                assert!(
                    (mag - 1.0).abs() < 1e-5,
                    "Impulse FFT at [{},{}] should have magnitude 1, got {}",
                    r,
                    c,
                    mag
                );
            }
        }
    }

    #[test]
    fn test_fft2d_parseval() {
        // Parseval's theorem: sum of |x|^2 = (1/N) * sum of |X|^2
        // For unnormalized FFT: sum of |x|^2 = (1/(N*M)) * sum of |X|^2
        let input = random_matrix_f32(8, 8, 42);
        let (fft_row, fft_col, _, _) = create_fft_plans_f32(8, 8);

        let output = fft2d(input.view(), &fft_row, &fft_col);

        let energy_spatial: f32 = input.iter().map(|x| x * x).sum();
        let energy_freq: f32 = output.iter().map(|x| x.norm_sqr()).sum();

        let expected_freq_energy = energy_spatial * 64.0; // N*M scaling for unnormalized FFT

        assert!(
            (energy_freq - expected_freq_energy).abs() / expected_freq_energy < 1e-4,
            "Parseval's theorem violated: spatial={}, freq={}, expected={}",
            energy_spatial,
            energy_freq,
            expected_freq_energy
        );
    }

    // ==================== FFT Edge Case Tests ====================

    #[test]
    fn test_fft2d_single_element() {
        let mut input = Array2::<f32>::zeros((1, 1));
        input[[0, 0]] = 2.71; // Euler's number approximation (not PI to avoid clippy)
        let (fft_row, fft_col, ifft_row, ifft_col) = create_fft_plans_f32(1, 1);

        let freq = fft2d(input.view(), &fft_row, &fft_col);
        assert!(
            (freq[[0, 0]].re - 2.71).abs() < 1e-5,
            "1x1 FFT should preserve value"
        );

        let output = ifft2d(&freq, &ifft_row, &ifft_col);
        assert!((output[[0, 0]] - 2.71).abs() < 1e-5, "1x1 roundtrip failed");
    }

    #[test]
    fn test_fft2d_non_square() {
        // Test non-square matrices
        let sizes = [(4, 8), (8, 4), (2, 16), (16, 2)];

        for (rows, cols) in sizes {
            let input = random_matrix_f32(rows, cols, (rows * 100 + cols) as u64);
            let (fft_row, fft_col, ifft_row, ifft_col) = create_fft_plans_f32(rows, cols);

            let freq = fft2d(input.view(), &fft_row, &fft_col);
            let output = ifft2d(&freq, &ifft_row, &ifft_col);

            assert!(
                arrays_approx_equal_f32(&input, &output, 1e-5),
                "Non-square {}x{} roundtrip failed",
                rows,
                cols
            );
        }
    }

    #[test]
    fn test_fft2d_large_values() {
        // Test numerical stability with large values
        let mut input = Array2::<f32>::zeros((8, 8));
        for r in 0..8 {
            for c in 0..8 {
                input[[r, c]] = ((r * 8 + c) as f32) * 1e5;
            }
        }
        let (fft_row, fft_col, ifft_row, ifft_col) = create_fft_plans_f32(8, 8);

        let freq = fft2d(input.view(), &fft_row, &fft_col);
        let output = ifft2d(&freq, &ifft_row, &ifft_col);

        // Use relative tolerance for large values
        for r in 0..8 {
            for c in 0..8 {
                let diff = (input[[r, c]] - output[[r, c]]).abs();
                let rel_err = diff / (input[[r, c]].abs() + 1e-10);
                assert!(
                    rel_err < 1e-4,
                    "Large value roundtrip failed at [{},{}]: input={}, output={}, rel_err={}",
                    r,
                    c,
                    input[[r, c]],
                    output[[r, c]],
                    rel_err
                );
            }
        }
    }

    #[test]
    fn test_fft2d_small_values() {
        // Test numerical stability with small values
        let mut input = Array2::<f32>::zeros((8, 8));
        for r in 0..8 {
            for c in 0..8 {
                input[[r, c]] = ((r * 8 + c) as f32 + 1.0) * 1e-6;
            }
        }
        let (fft_row, fft_col, ifft_row, ifft_col) = create_fft_plans_f32(8, 8);

        let freq = fft2d(input.view(), &fft_row, &fft_col);
        let output = ifft2d(&freq, &ifft_row, &ifft_col);

        assert!(
            arrays_approx_equal_f32(&input, &output, 1e-10),
            "Small value roundtrip failed"
        );
    }

    // ==================== WHT Round-Trip Tests (f32) ====================

    #[test]
    fn test_wht_8x8_roundtrip() {
        let input = random_matrix_f32(8, 8, 54321);

        let freq = wht2d_8x8_forward(input.view());
        let output = wht2d_8x8_inverse(&freq);

        assert!(
            arrays_approx_equal_f32(&input, &output, 1e-6),
            "WHT roundtrip failed: max diff = {}",
            input
                .iter()
                .zip(output.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        );
    }

    #[test]
    fn test_wht_8x8_roundtrip_multiple() {
        // Test with 10 different random inputs
        for seed in 0..10u64 {
            let input = random_matrix_f32(8, 8, seed * 13331);

            let freq = wht2d_8x8_forward(input.view());
            let output = wht2d_8x8_inverse(&freq);

            let max_diff = input
                .iter()
                .zip(output.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(
                arrays_approx_equal_f32(&input, &output, 1e-6),
                "WHT roundtrip failed for seed {}: max diff = {}",
                seed,
                max_diff
            );
        }
    }

    // ==================== WHT f64 Round-Trip Tests ====================

    #[test]
    fn test_wht_8x8_roundtrip_f64() {
        let input = random_matrix_f64(8, 8, 54321);

        let freq = wht2d_8x8_forward(input.view());
        let output = wht2d_8x8_inverse(&freq);

        assert!(
            arrays_approx_equal_f64(&input, &output, 1e-14),
            "WHT f64 roundtrip failed: max diff = {}",
            input
                .iter()
                .zip(output.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max)
        );
    }

    #[test]
    fn test_wht_8x8_roundtrip_multiple_f64() {
        for seed in 0..10u64 {
            let input = random_matrix_f64(8, 8, seed * 13331);

            let freq = wht2d_8x8_forward(input.view());
            let output = wht2d_8x8_inverse(&freq);

            let max_diff = input
                .iter()
                .zip(output.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);

            assert!(
                arrays_approx_equal_f64(&input, &output, 1e-14),
                "WHT f64 roundtrip failed for seed {}: max diff = {}",
                seed,
                max_diff
            );
        }
    }

    // ==================== WHT Known-Value Tests ====================

    #[test]
    fn test_wht_zeros() {
        let input = Array2::<f32>::zeros((8, 8));
        let output = wht2d_8x8_forward(input.view());

        for val in output.iter() {
            assert!(
                val.norm() < 1e-10,
                "WHT of zeros should be zeros, got magnitude {}",
                val.norm()
            );
        }
    }

    #[test]
    fn test_wht_constant() {
        // All ones: First coefficient (DC) = 8*8 = 64, rest = 0
        // (WHT of constant is impulse in first coefficient)
        let input = Array2::<f32>::ones((8, 8));
        let output = wht2d_8x8_forward(input.view());

        // DC component should be 64 (sum of all 64 ones)
        let dc = output[[0, 0]];
        assert!(
            (dc.re - 64.0).abs() < 1e-6,
            "WHT DC component should be 64, got {}",
            dc.re
        );

        // All other components should be 0
        for r in 0..8 {
            for c in 0..8 {
                if r != 0 || c != 0 {
                    let val = output[[r, c]];
                    assert!(
                        val.norm() < 1e-6,
                        "Non-DC component [{},{}] should be 0, got {}",
                        r,
                        c,
                        val.norm()
                    );
                }
            }
        }
    }

    #[test]
    fn test_wht_impulse() {
        // Single 1.0 at (0,0), rest zeros
        // WHT of impulse at origin should spread to all coefficients
        let mut input = Array2::<f32>::zeros((8, 8));
        input[[0, 0]] = 1.0;

        let output = wht2d_8x8_forward(input.view());

        // All WHT coefficients should have the same value
        let expected = output[[0, 0]].re;
        for r in 0..8 {
            for c in 0..8 {
                assert!(
                    (output[[r, c]].re - expected).abs() < 1e-6,
                    "WHT of impulse should have uniform coefficients, got [{},{}]={}",
                    r,
                    c,
                    output[[r, c]].re
                );
            }
        }
    }

    #[test]
    fn test_wht_symmetry() {
        // WHT is self-inverse (up to scaling)
        // Applying it twice should give back scaled input
        let input = random_matrix_f32(8, 8, 99999);

        let once = wht2d_8x8_forward(input.view());
        // Convert Complex output to f32 for second forward pass
        let mut once_real = Array2::<f32>::zeros((8, 8));
        for r in 0..8 {
            for c in 0..8 {
                once_real[[r, c]] = once[[r, c]].re;
            }
        }
        let twice = wht2d_8x8_forward(once_real.view());

        // Should be 64 * input (WHT is self-inverse with scale N)
        for r in 0..8 {
            for c in 0..8 {
                let expected = input[[r, c]] * 64.0;
                let actual = twice[[r, c]].re;
                assert!(
                    (expected - actual).abs() < 1e-4,
                    "WHT symmetry failed at [{},{}]: expected {}, got {}",
                    r,
                    c,
                    expected,
                    actual
                );
            }
        }
    }

    // ==================== WHT Edge Case Tests ====================

    #[test]
    fn test_wht_near_zero() {
        // Test numerical stability with very small values
        let mut input = Array2::<f32>::zeros((8, 8));
        let mut rng = SimpleLcg::new(11111);
        for r in 0..8 {
            for c in 0..8 {
                input[[r, c]] = rng.next_f32() * 1e-10;
            }
        }

        let freq = wht2d_8x8_forward(input.view());
        let output = wht2d_8x8_inverse(&freq);

        assert!(
            arrays_approx_equal_f32(&input, &output, 1e-14),
            "WHT near-zero roundtrip failed"
        );
    }

    #[test]
    fn test_wht_large_values() {
        // Test numerical stability with large values
        let mut input = Array2::<f32>::zeros((8, 8));
        for r in 0..8 {
            for c in 0..8 {
                input[[r, c]] = ((r * 8 + c) as f32 + 1.0) * 1e5;
            }
        }

        let freq = wht2d_8x8_forward(input.view());
        let output = wht2d_8x8_inverse(&freq);

        // Use relative tolerance for large values
        for r in 0..8 {
            for c in 0..8 {
                let diff = (input[[r, c]] - output[[r, c]]).abs();
                let rel_err = diff / (input[[r, c]].abs() + 1e-10);
                assert!(
                    rel_err < 1e-5,
                    "WHT large value roundtrip failed at [{},{}]: rel_err={}",
                    r,
                    c,
                    rel_err
                );
            }
        }
    }

    #[test]
    fn test_wht_alternating_pattern() {
        // Test with checkerboard pattern (alternating +1, -1)
        let mut input = Array2::<f32>::zeros((8, 8));
        for r in 0..8 {
            for c in 0..8 {
                input[[r, c]] = if (r + c) % 2 == 0 { 1.0 } else { -1.0 };
            }
        }

        let freq = wht2d_8x8_forward(input.view());
        let output = wht2d_8x8_inverse(&freq);

        assert!(
            arrays_approx_equal_f32(&input, &output, 1e-6),
            "WHT alternating pattern roundtrip failed"
        );
    }

    #[test]
    fn test_wht_imaginary_part_ignored() {
        // Verify that inverse WHT only uses real part of input
        let input = random_matrix_f32(8, 8, 77777);
        let freq = wht2d_8x8_forward(input.view());

        // Add imaginary components that should be ignored
        let mut freq_with_imag = freq.clone();
        for r in 0..8 {
            for c in 0..8 {
                freq_with_imag[[r, c]] = Complex::new(freq[[r, c]].re, 999.0);
            }
        }

        let output_clean = wht2d_8x8_inverse(&freq);
        let output_imag = wht2d_8x8_inverse(&freq_with_imag);

        assert!(
            arrays_approx_equal_f32(&output_clean, &output_imag, 1e-10),
            "WHT inverse should ignore imaginary part"
        );
    }
}
