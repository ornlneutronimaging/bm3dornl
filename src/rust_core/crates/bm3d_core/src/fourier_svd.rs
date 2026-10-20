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
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

fn power_iteration_k1<F: Bm3dFloat>(
    matrix: ArrayView2<F>,
    max_iter: usize,
    _tol: F,
) -> (Array1<F>, F, Array1<F>) {
    let (rows, cols) = matrix.dim();

    let init_val = F::one() / F::from_f64_c((cols as f64).sqrt());
    let mut v = Array1::from_elem(cols, init_val);

    let mut u = Array1::zeros(rows);
    let mut s = F::zero();
    let epsilon = F::from_f64_c(1e-10);

    for _ in 0..max_iter {
        u = matrix.dot(&v);

        let u_norm = u.dot(&u).sqrt();
        if u_norm < epsilon {
            break;
        }
        u.mapv_inplace(|x| x / u_norm);

        v = matrix.t().dot(&u);

        let v_norm = v.dot(&v).sqrt();
        s = v_norm;

        if v_norm < epsilon {
            break;
        }
        v.mapv_inplace(|x| x / v_norm);
    }

    (u, s, v)
}

fn compute_vertical_energy_profile<F: Bm3dFloat>(
    sinogram: ArrayView2<F>,
    notch_width: F,
) -> Array1<F> {
    let (rows, cols) = sinogram.dim();

    let mut planner = FftPlanner::<F>::new();
    let fft_row = planner.plan_fft_forward(cols);
    let fft_col = planner.plan_fft_forward(rows);
    let ifft_row = planner.plan_fft_inverse(cols);
    let ifft_col = planner.plan_fft_inverse(rows);

    let freq_domain = transforms::fft2d(sinogram, &fft_row, &fft_col);

    let mut filtered_freq = freq_domain;

    let neg_half = F::from_f64_c(-0.5);
    let sigma_sq = notch_width * notch_width;
    let rows_f = F::usize_as(rows);
    let rows_half = rows_f / F::from_f64_c(2.0);

    let cols_f = F::usize_as(cols);
    let cols_half = cols_f / F::from_f64_c(2.0);
    let mut x_weights = Vec::with_capacity(cols);
    for c in 0..cols {
        let c_f = F::usize_as(c);
        let dist = if c_f <= cols_half { c_f } else { cols_f - c_f };
        let dist_sq = dist * dist;
        let low_pass = (neg_half * dist_sq / sigma_sq).exp();
        x_weights.push(F::one() - low_pass);
    }

    for r in 0..rows {
        let r_f = F::usize_as(r);
        let dist = if r_f <= rows_half { r_f } else { rows_f - r_f };

        let dist_sq = dist * dist;
        let y_weight = (neg_half * dist_sq / sigma_sq).exp();

        for c in 0..cols {
            let w_val = y_weight * x_weights[c];
            let w_complex = Complex::new(w_val, F::zero());
            filtered_freq[[r, c]] *= w_complex;
        }
    }

    let spatial_filtered = transforms::ifft2d(&filtered_freq, &ifft_row, &ifft_col);

    let mut energy_profile = Array1::<F>::zeros(cols);
    let rows_f_inv = F::one() / rows_f;

    for c in 0..cols {
        let mut sum_abs = F::zero();
        for r in 0..rows {
            sum_abs += spatial_filtered[[r, c]].abs();
        }
        energy_profile[c] = sum_abs * rows_f_inv;
    }

    let mut energy_vec: Vec<F> = energy_profile.to_vec();
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

const NOOP_THRESHOLD_RATIO: f64 = 1e-6;

const FLOOR_RATIO: f64 = 1e-6;

const MIN_INFORMATIVE_FRACTION: f64 = 0.25;
const MIN_INFORMATIVE_COLUMNS: usize = 16;

const MAX_STREAK_ENERGY_FRACTION: f64 = 0.1;

fn gate_v_streak<F: Bm3dFloat>(
    v_detail: &Array1<F>,
    base_thresh: F,
    modulator: Option<&Array1<F>>,
) -> Array1<F> {
    let exponent = 6;
    let mut v_streak = Array1::<F>::zeros(v_detail.len());

    for c in 0..v_detail.len() {
        let x = v_detail[c];
        let thresh = if let Some(m) = modulator {
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

    v_streak
}

pub fn fourier_svd_removal<F: Bm3dFloat>(
    sinogram: ArrayView2<F>,
    fft_alpha: F,
    notch_width: F,
) -> Array2<F> {
    let (rows, cols) = sinogram.dim();
    if rows == 0 || cols == 0 {
        return sinogram.to_owned();
    }

    let (u, s, v) = power_iteration_k1(sinogram, 20, F::from_f64_c(1e-6));

    let v_slice = v.as_slice().unwrap();
    let v_smooth_vec = compute_1d_median_filter(v_slice, 51);
    let v_smooth = Array1::from(v_smooth_vec);
    let v_detail = &v - &v_smooth;

    let modulator = if fft_alpha > F::from_f64_c(1e-6) {
        let energy = compute_vertical_energy_profile(sinogram, notch_width);
        Some(energy.mapv(|e| F::one() + fft_alpha * e))
    } else {
        None
    };

    let vals: Vec<f64> = v_detail.iter().map(|x| x.to_f64().unwrap_or(0.0)).collect();
    let mut sorted_vals = vals.clone();
    sorted_vals.sort_by(|a, b| a.total_cmp(b));
    let median = sorted_vals[vals.len() / 2];
    let mut dev: Vec<f64> = vals.iter().map(|x| (x - median).abs()).collect();
    dev.sort_by(|a, b| a.total_cmp(b));
    let raw_mad_sigma = dev[dev.len() / 2] / 0.6745;
    let max_dev = dev[dev.len() - 1];

    if 3.0 * raw_mad_sigma < NOOP_THRESHOLD_RATIO * max_dev {
        if raw_mad_sigma == 0.0 {
            return sinogram.to_owned();
        }

        let floor_cut = FLOOR_RATIO * max_dev;
        let nonfloor: Vec<F> = (0..cols)
            .filter(|&c| (vals[c] - median).abs() > floor_cut)
            .map(|c| v_detail[c])
            .collect();

        let needed =
            MIN_INFORMATIVE_COLUMNS.max((MIN_INFORMATIVE_FRACTION * cols as f64).ceil() as usize);
        if nonfloor.len() < needed {
            // Too few informative columns to estimate a streak scale from.
            return sinogram.to_owned();
        }

        let sigma_rescue = estimate_robust_sigma(Array1::from(nonfloor).view());
        let base_thresh = F::from_f64_c(sigma_rescue * 3.0);
        let v_streak = gate_v_streak(&v_detail, base_thresh, modulator.as_ref());

        let streak_energy = s.to_f64().unwrap_or(0.0)
            * v_streak
                .iter()
                .map(|x| {
                    let xf = x.to_f64().unwrap_or(0.0);
                    xf * xf
                })
                .sum::<f64>()
                .sqrt();
        let data_energy = sinogram
            .iter()
            .map(|x| {
                let xf = x.to_f64().unwrap_or(0.0);
                xf * xf
            })
            .sum::<f64>()
            .sqrt();
        if streak_energy > MAX_STREAK_ENERGY_FRACTION * data_energy {
            // A correction this large would be removing the sample.
            return sinogram.to_owned();
        }

        let scaled_u = u.mapv(|x| x * s);
        let mut corrected = sinogram.to_owned();
        for r in 0..rows {
            let u_val = scaled_u[r];
            for c in 0..cols {
                corrected[[r, c]] -= u_val * v_streak[c];
            }
        }
        return corrected;
    }

    let sigma = estimate_robust_sigma(v_detail.view());
    let base_thresh = F::from_f64_c(sigma * 3.0);

    let v_streak = gate_v_streak(&v_detail, base_thresh, modulator.as_ref());

    let scaled_u = u.mapv(|x| x * s);

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Rank-1 sinogram: outer(profile-with-row-variation, base).
    fn rank1_sinogram(rows: usize, base: &[f64]) -> Array2<f64> {
        Array2::from_shape_fn((rows, base.len()), |(r, c)| {
            (1.0 + 0.1 * (r as f64 / rows as f64)) * base[c]
        })
    }

    /// Deterministic jitter in [-1, 1], no RNG dependency.
    fn jitter(c: usize) -> f64 {
        ((c as f64 * 12.9898).sin() * 43758.5453).fract()
    }

    #[test]
    fn flat_air_majority_is_rescued() {
        let cols = 200;
        let air = 60; // 120 of 200 columns are near-flat air
        let base: Vec<f64> = (0..cols)
            .map(|c| {
                if c < air || c >= cols - air {
                    1.0e-8 * (1.0 + 1.0e-3 * jitter(c))
                } else {
                    // smooth sample plus per-column gain jitter (the streaks)
                    0.5 * (1.0 + 0.02 * jitter(c))
                }
            })
            .collect();
        let sino = rank1_sinogram(64, &base);

        let corrected = fourier_svd_removal(sino.view(), 1.0, 2.0);

        let max_change = sino
            .iter()
            .zip(corrected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_change > 1.0e-4,
            "flat-air sinogram was returned unchanged: max|change| = {max_change}"
        );

        let sample_mean_in: f64 = sino.column(cols / 2).iter().sum::<f64>() / sino.nrows() as f64;
        let sample_mean_out: f64 =
            corrected.column(cols / 2).iter().sum::<f64>() / corrected.nrows() as f64;
        assert!(
            (sample_mean_in - sample_mean_out).abs() < 0.05 * sample_mean_in,
            "sample level moved: {sample_mean_in} -> {sample_mean_out}"
        );
    }

    #[test]
    fn rescue_declines_on_too_few_informative_columns() {
        let cols = 200;
        let base: Vec<f64> = (0..cols)
            .map(|c| {
                if (90..110).contains(&c) {
                    0.5 * (1.0 + 0.02 * jitter(c))
                } else {
                    1.0e-8 * (1.0 + 1.0e-3 * jitter(c))
                }
            })
            .collect();
        let sino = rank1_sinogram(64, &base);

        let corrected = fourier_svd_removal(sino.view(), 1.0, 2.0);

        assert!(
            sino.iter().zip(corrected.iter()).all(|(a, b)| a == b),
            "rescue acted despite having only ~20 informative columns"
        );
    }

    #[test]
    fn rescue_declines_on_structure_scale_correction() {
        let cols = 200;
        let base: Vec<f64> = (0..cols)
            .map(|c| {
                if c < 120 {
                    1.0e-8 * (1.0 + 1.0e-3 * jitter(c))
                } else if c % 2 == 0 {
                    // high-contrast alternation: structure-scale detail
                    0.5
                } else {
                    1.5
                }
            })
            .collect();
        let sino = rank1_sinogram(64, &base);

        let corrected = fourier_svd_removal(sino.view(), 1.0, 2.0);

        assert!(
            sino.iter().zip(corrected.iter()).all(|(a, b)| a == b),
            "rescue applied a structure-scale correction"
        );
    }

    #[test]
    fn zero_dispersion_returns_input_unchanged() {
        let cols = 200;
        let base: Vec<f64> = (0..cols)
            .map(|c| {
                if c < 150 {
                    1.0e-8 // exactly flat: detail exactly zero
                } else {
                    0.5 * (1.0 + 0.02 * jitter(c))
                }
            })
            .collect();
        let sino = rank1_sinogram(64, &base);

        let corrected = fourier_svd_removal(sino.view(), 1.0, 2.0);

        assert!(
            sino.iter().zip(corrected.iter()).all(|(a, b)| a == b),
            "zero-dispersion input was altered"
        );
    }

    #[test]
    fn empty_input_is_returned_unchanged() {
        let sino = Array2::<f64>::zeros((0, 0));
        let corrected = fourier_svd_removal(sino.view(), 1.0, 2.0);
        assert_eq!(corrected.dim(), (0, 0));
    }
}
