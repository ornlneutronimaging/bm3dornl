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

/// The rescue path below exists for inputs where the MAD-derived threshold is
/// numerical silence. It fires when the threshold falls under this fraction of
/// the largest deviation, i.e. when the gate could remove nothing measurable
/// and the whole call would return its input (issue #133). Measured margins:
/// the degenerate flat-background phantoms sit near 5e-8; 533 of 540 real
/// CG-1D slices sit from about 2.7e-2 up to 5e-1 — more than four decades
/// either side of the cut — and the remaining 7 have exactly zero dispersion,
/// handled separately below.
const NOOP_THRESHOLD_RATIO: f64 = 1e-6;

/// Deviations at or below this fraction of the largest deviation are the
/// degenerate floor itself (numerically silent entries), excluded when
/// re-estimating the scale. Only consulted inside the rescue branch, where the
/// floor is machine-precision residue; genuine noise populations sit orders of
/// magnitude above it.
const FLOOR_RATIO: f64 = 1e-6;

/// The rescue declines unless at least this fraction of columns (and at least
/// [`MIN_INFORMATIVE_COLUMNS`]) carry information above the floor. With fewer,
/// streaks cannot be told apart from structure, and a no-op is preferable to
/// guessing.
const MIN_INFORMATIVE_FRACTION: f64 = 0.25;
const MIN_INFORMATIVE_COLUMNS: usize = 16;

/// The rescue declines if the correction it would apply carries more than this
/// fraction of the input's total energy. Genuine streak removal is a small
/// correction (measured 1.7% on the benchmark phantom); a rank-1 subtraction
/// approaching the data's own scale is removing the sample, not streaks.
const MAX_STREAK_ENERGY_FRACTION: f64 = 0.1;

/// Apply the magnitude gate: mask = 1 / (1 + (|x|/thresh)^6), streak = x * mask.
///
/// Detail below the threshold is treated as streak (removed by the caller);
/// detail above it is protected as structure.
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
///
/// The gate keeps detail below its threshold as streak and protects detail
/// above it as structure, so the threshold must land between the streak and
/// structure amplitudes. It is 3x a MAD-based scale of `v_detail`. MAD is the
/// median of the absolute deviations, and on a sinogram whose columns mostly
/// see flat air the majority of those deviations are numerically zero, so MAD
/// collapses to machine residue and the gate removes nothing at all — the
/// whole call returns its input (issue #133). When that specific degeneracy is
/// detected, the scale is re-estimated over the entries above the numerical
/// floor; the rescue is guarded so it declines (keeping the no-op) rather than
/// ever applying a structure-scale correction. Inputs where the threshold is
/// measurable — all real measured data checked — take the original path
/// unchanged.
pub fn fourier_svd_removal<F: Bm3dFloat>(
    sinogram: ArrayView2<F>,
    fft_alpha: F,
    notch_width: F,
) -> Array2<F> {
    let (rows, cols) = sinogram.dim();
    if rows == 0 || cols == 0 {
        return sinogram.to_owned();
    }

    // 1. Power Iteration (K=1)
    let (u, s, v) = power_iteration_k1(sinogram, 20, F::from_f64_c(1e-6));

    // 2. Filter v (Horizontal Profile)
    let v_slice = v.as_slice().unwrap(); // Assuming standard layout, safe for owned arrays usually.
    // If strided, map to vec.
    // v is owned Array1, so it is contiguous.
    let v_smooth_vec = compute_1d_median_filter(v_slice, 51);
    let v_smooth = Array1::from(v_smooth_vec);
    let v_detail = &v - &v_smooth;

    // Compute Threshold Modulator
    let modulator = if fft_alpha > F::from_f64_c(1e-6) {
        let energy = compute_vertical_energy_profile(sinogram, notch_width);
        // mod = 1 + alpha * energy
        Some(energy.mapv(|e| F::one() + fft_alpha * e))
    } else {
        None
    };

    // Degeneracy check on the RAW MAD, before estimate_robust_sigma's
    // mad == 0 fallback can substitute a standard deviation: that fallback is
    // reached from the same flat-background degeneracy and its value is set by
    // whatever structure is present, which is exactly the scale the gate must
    // never adopt.
    let vals: Vec<f64> = v_detail.iter().map(|x| x.to_f64().unwrap_or(0.0)).collect();
    let mut sorted_vals = vals.clone();
    sorted_vals.sort_by(|a, b| a.total_cmp(b));
    let median = sorted_vals[vals.len() / 2];
    let mut dev: Vec<f64> = vals.iter().map(|x| (x - median).abs()).collect();
    dev.sort_by(|a, b| a.total_cmp(b));
    let raw_mad_sigma = dev[dev.len() / 2] / 0.6745;
    let max_dev = dev[dev.len() - 1];

    if 3.0 * raw_mad_sigma < NOOP_THRESHOLD_RATIO * max_dev {
        // The threshold is numerical silence: the gate below would remove
        // nothing and the call would return its input. Try to rescue.

        // Exactly zero dispersion is different from silently small dispersion.
        // The median filter returns an element of its window, so on a smooth
        // column profile more than half of the detail entries can be exactly
        // zero — measured on 7 of 540 sinograms of the real CG-1D volume, whose
        // zeros are median-filter coincidences on real detector columns, not
        // flat air. No scale can be estimated from a zero median deviation, and
        // every substitute measured worse than leaving the data alone (the old
        // mad == 0 standard-deviation fallback degraded those slices; so did
        // rescuing them). Return the input unchanged.
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

        // Genuine streak removal is a small correction. ||u|| = 1, so the
        // rank-1 correction's Frobenius norm is s * ||v_streak||.
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

    // 3. Magnitude Gating (original path, unchanged)
    let sigma = estimate_robust_sigma(v_detail.view());
    let base_thresh = F::from_f64_c(sigma * 3.0);

    let v_streak = gate_v_streak(&v_detail, base_thresh, modulator.as_ref());

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

    /// Regression for issue #133: a sinogram whose columns mostly see flat air
    /// must still have its streaks removed, not be returned unchanged.
    ///
    /// The air carries a minuscule per-column wobble so its detail is tiny but
    /// not exactly zero: that is sino.npy's regime, where MAD collapses without
    /// tripping the old `mad == 0` fallback and the previous code returned its
    /// input untouched. This test fails against that code.
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

        // The smooth sample level must survive; only the jitter is removable.
        let sample_mean_in: f64 = sino.column(cols / 2).iter().sum::<f64>() / sino.nrows() as f64;
        let sample_mean_out: f64 =
            corrected.column(cols / 2).iter().sum::<f64>() / corrected.nrows() as f64;
        assert!(
            (sample_mean_in - sample_mean_out).abs() < 0.05 * sample_mean_in,
            "sample level moved: {sample_mean_in} -> {sample_mean_out}"
        );
    }

    /// With too few informative columns the rescue must decline and return the
    /// input exactly, never estimate a scale from a handful of entries. The air
    /// carries a tiny wobble so dispersion is nonzero and the count guard, not
    /// the zero-dispersion rule, is what declines.
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

    /// A rescue whose correction would carry a large share of the data's
    /// energy is removing the sample, not streaks, and must decline. Wobbled
    /// air keeps dispersion nonzero so the energy guard is what declines.
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

    /// Exactly zero dispersion: more than half the detail entries are exactly
    /// zero (median-filter coincidences on a smooth profile), yet enough
    /// informative columns exist that the count guard alone would let a rescue
    /// proceed. No scale is estimable from a zero median deviation, so the
    /// input must come back unchanged. Fails on the previous code, whose
    /// mad == 0 fallback substituted a standard deviation and altered the data.
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
