//! # Fourier-SVD Streak Removal
//!
//! A two-stage algorithm for removing vertical streak artifacts from sinograms:
//!
//! ## Stage 1: FFT-Guided Energy Detection
//! - Apply FFT and isolate vertical frequencies (Fy ≈ 0) using Gaussian notch filter
//! - Compute per-column energy profile from isolated streak spectrum
//! - Use energy profile to spatially modulate removal threshold
//!
//! ## Stage 2: Rank-1 SVD with Magnitude Gating
//! - Extract first principal component via power iteration
//! - Median filter the v-vector to separate baseline from streak detail
//! - Apply soft magnitude gating: `1 / (1 + (|v| / threshold)^exponent)`
//! - Reconstruct streak as rank-1 outer product and subtract from input
//!
//! ## Parameters
//! - `fft_alpha`: Controls FFT energy influence on threshold modulation (default: 1.0)
//! - `notch_width`: Gaussian notch filter width in frequency domain (default: 2.0)

use crate::float_trait::Bm3dFloat;
use crate::transforms;
use crate::utils::{compute_1d_median_filter, estimate_robust_sigma};
use ndarray::{Array1, Array2, ArrayView2};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Power Iteration to find the First Principal Component (K=1).
/// Returns (u, s, v_t) for the largest singular value.
/// Input A: (rows, cols)
/// u: (rows,)
/// s: scalar
/// v: (cols,) - Note: this is v, not v_t in the sense of V^T row.
fn power_iteration_k1<F: Bm3dFloat>(
    matrix: ArrayView2<F>,
    max_iter: usize,
    _tol: F,
) -> (Array1<F>, F, Array1<F>) {
    let (rows, cols) = matrix.dim();

    // Random initialization for v
    let init_val = F::one() / F::from_f64_c((cols as f64).sqrt());
    let mut v = Array1::from_elem(cols, init_val);

    let mut u = Array1::zeros(rows);
    let mut s = F::zero();
    let epsilon = F::from_f64_c(1e-10);

    for _ in 0..max_iter {
        // u = A * v
        u = matrix.dot(&v);

        // Normalize u
        let u_norm = u.dot(&u).sqrt();
        if u_norm < epsilon {
            break;
        }
        u.mapv_inplace(|x| x / u_norm);

        // v = A^T * u
        v = matrix.t().dot(&u);

        // Sigma = norm(v_unnormalized)
        let v_norm = v.dot(&v).sqrt();
        s = v_norm;

        if v_norm < epsilon {
            break;
        }
        v.mapv_inplace(|x| x / v_norm);
    }

    (u, s, v)
}

/// Compute Vertical Energy Profile using FFT Notch.
///
/// Returns a 1D array of length `cols` representing the vertical energy probability.
///
/// 1. FFT2D
/// 2. Apply Gaussian Notch at Fy=0 (keep only vertical frequencies)
/// 3. IFFT2D
/// 4. Mean Absolute Value along rows
fn compute_vertical_energy_profile<F: Bm3dFloat>(
    sinogram: ArrayView2<F>,
    notch_width: F,
) -> Array1<F> {
    let (rows, cols) = sinogram.dim();

    // Create FFT plans locally (expensive but robust)
    let mut planner = FftPlanner::<F>::new();
    let fft_row = planner.plan_fft_forward(cols);
    let fft_col = planner.plan_fft_forward(rows);
    let ifft_row = planner.plan_fft_inverse(cols);
    let ifft_col = planner.plan_fft_inverse(rows);

    // 1. FFT
    let freq_domain = transforms::fft2d(sinogram, &fft_row, &fft_col);

    // 2. Gaussian Notch (Bandpass at Fy=0)
    // Filter = exp( - (y - cy)^2 / (2 * sigma^2) )
    // We want to KEEP Fy ~ 0.
    // Frequencies are shifted? transforms::fft2d output is standard FFT layout (DC at 0,0)
    // So Fy=0 corresponds to indices close to 0 and close to N.
    // We need to handle wrapping indices for valid distance calculation.

    let mut filtered_freq = freq_domain; // Move to mutable

    let neg_half = F::from_f64_c(-0.5);
    let sigma_sq = notch_width * notch_width;
    let rows_f = F::usize_as(rows);
    let rows_half = rows_f / F::from_f64_c(2.0);

    // We operate on unshifted FFT data.
    // Index r corresponds to frequency:
    // if r < rows/2: f = r
    // else: f = r - rows
    // Distance to 0 is just min(r, rows-r).

    // Precompute column weights (High Pass in X to suppress wide structures/DC)
    let cols_f = F::usize_as(cols);
    let cols_half = cols_f / F::from_f64_c(2.0);
    let mut x_weights = Vec::with_capacity(cols);
    for c in 0..cols {
        let c_f = F::usize_as(c);
        let dist = if c_f <= cols_half { c_f } else { cols_f - c_f };
        let dist_sq = dist * dist;
        // High Pass: 1.0 - LowPass. We use same sigma for simplicity or slightly wider?
        // Let's use same sigma to reject 'streak-like' low freq structures.
        let low_pass = (neg_half * dist_sq / sigma_sq).exp();
        x_weights.push(F::one() - low_pass);
    }

    for r in 0..rows {
        let r_f = F::usize_as(r);
        let dist = if r_f <= rows_half { r_f } else { rows_f - r_f };

        let dist_sq = dist * dist;
        let y_weight = (neg_half * dist_sq / sigma_sq).exp(); // Low Pass Y (Keep Vertical)

        for c in 0..cols {
            // Combined weight: Keep Vertical AND Keep High Freq X
            let w_val = y_weight * x_weights[c];
            let w_complex = Complex::new(w_val, F::zero());
            filtered_freq[[r, c]] *= w_complex;
        }
    }

    // 3. IFFT
    let spatial_filtered = transforms::ifft2d(&filtered_freq, &ifft_row, &ifft_col);

    // 4. Mean Mean Absolute along Rows (Vertical Project)
    let mut energy_profile = Array1::<F>::zeros(cols);
    let rows_f_inv = F::one() / rows_f;

    for c in 0..cols {
        let mut sum_abs = F::zero();
        for r in 0..rows {
            sum_abs += spatial_filtered[[r, c]].abs();
        }
        energy_profile[c] = sum_abs * rows_f_inv;
    }

    // Normalize profile?
    // Python code: normalized by Median.
    // "med_energy = np.median(energy_profile)"
    // "norm_profile = energy_profile / med_energy"

    // We need median of energy_profile
    let mut energy_vec: Vec<F> = energy_profile.to_vec();
    // Sort to find median
    // F doesn't implement Ord, so partial_cmp
    energy_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let len = energy_vec.len();
    let median = if len > 0 {
        if len % 2 == 1 {
            energy_vec[len / 2]
        } else {
            (energy_vec[len / 2 - 1] + energy_vec[len / 2]) * F::from_f64_c(0.5)
        }
    } else {
        F::one()
    };

    if median > F::from_f64_c(1e-10) {
        let inv_med = F::one() / median;
        energy_profile.mapv_inplace(|x| x * inv_med);
    }

    energy_profile
}

/// Fourier-SVD Streak Removal
///
/// A two-stage algorithm combining FFT-based energy detection with rank-1 SVD:
///
/// 1. SVD(A) -> u, s, v (First principal component via power iteration)
/// 2. v_smooth = MedianFilter(v)
/// 3. v_detail = v - v_smooth
/// 4. v_streak = Gate(v_detail, thresh) where thresh is modulated by FFT energy if alpha > 0
/// 5. StreakImage = u * s * v_streak^T
/// 6. Corrected = A - StreakImage
pub fn fourier_svd_removal<F: Bm3dFloat>(
    sinogram: ArrayView2<F>,
    fft_alpha: F,
    notch_width: F,
) -> Array2<F> {
    let (rows, cols) = sinogram.dim();

    // 1. Power Iteration (K=1)
    let (u, s, v) = power_iteration_k1(sinogram, 20, F::from_f64_c(1e-6));

    // 2. Filter v (Horizontal Profile)
    let v_slice = v.as_slice().unwrap(); // Assuming standard layout, safe for owned arrays usually.
                                         // If strided, map to vec.
                                         // v is owned Array1, so it is contiguous.
    let v_smooth_vec = compute_1d_median_filter(v_slice, 51);
    let v_smooth = Array1::from(v_smooth_vec);
    let v_detail = &v - &v_smooth;

    // 3. Magnitude Gating
    let sigma = estimate_robust_sigma(v_detail.view());
    let base_thresh = F::from_f64_c(sigma * 3.0);

    // Compute Threshold Modulator
    let modulator = if fft_alpha > F::from_f64_c(1e-6) {
        let energy = compute_vertical_energy_profile(sinogram, notch_width);
        // mod = 1 + alpha * energy
        Some(energy.mapv(|e| F::one() + fft_alpha * e))
    } else {
        None
    };

    // Gate function
    // mask = 1.0 / (1.0 + (abs(x)/thresh)^6)
    let exponent = 6;

    // We construct v_streak.
    // If modulator exists, thresh varies per column (c).
    let mut v_streak = Array1::<F>::zeros(cols);

    for c in 0..cols {
        let x = v_detail[c];
        let thresh = if let Some(ref m) = modulator {
            base_thresh * m[c]
        } else {
            base_thresh
        };

        let mask = if thresh > F::from_f64_c(1e-10) {
            let ratio = x.abs() / thresh;
            F::one() / (F::one() + ratio.powi(exponent))
        } else {
            F::zero()
        };

        v_streak[c] = x * mask;
    }

    // 4. Reconstruct
    // Streak = s * (u * v_streak^T)
    let scaled_u = u.mapv(|x| x * s);

    // Outer product
    let mut corrected = sinogram.to_owned();

    for r in 0..rows {
        let u_val = scaled_u[r];
        for c in 0..cols {
            let streak_val = u_val * v_streak[c];
            corrected[[r, c]] -= streak_val;
        }
    }

    corrected
}
