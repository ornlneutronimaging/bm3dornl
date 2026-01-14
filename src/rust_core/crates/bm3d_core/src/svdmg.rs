use crate::utils::{compute_1d_median_filter, estimate_robust_sigma};
use ndarray::{Array2, ArrayView2};

/// Power Iteration to find the First Principal Component (K=1).
/// Returns (u, s, v_t) for the largest singular value.
/// Input A: (rows, cols)
/// u: (rows,)
/// s: scalar
/// v: (cols,) - Note: this is v, not v_t in the sense of V^T row.
fn power_iteration_k1(
    matrix: ArrayView2<f64>,
    max_iter: usize,
    _tol: f64,
) -> (ndarray::Array1<f64>, f64, ndarray::Array1<f64>) {
    let (rows, cols) = matrix.dim();

    // Random initialization for v
    let mut v = ndarray::Array1::from_elem(cols, 1.0 / (cols as f64).sqrt());

    let mut u = ndarray::Array1::zeros(rows);
    let mut s = 0.0;

    for _ in 0..max_iter {
        // u = A * v
        u = matrix.dot(&v);

        // Normalize u
        let u_norm = u.dot(&u).sqrt();
        if u_norm < 1e-10 {
            break;
        }
        u.mapv_inplace(|x| x / u_norm);

        // v = A^T * u
        v = matrix.t().dot(&u);

        // Sigma = norm(v_unnormalized)
        // More accurately: s = u^T * A * v (if v is normalized)
        // Here v is A^T * u, so norm(v) IS sigma.
        let v_norm = v.dot(&v).sqrt();
        s = v_norm;

        if v_norm < 1e-10 {
            break;
        }
        v.mapv_inplace(|x| x / v_norm);

        // Check convergence (optional, but fixed iter is usually fine)
    }

    (u, s, v)
}

/// SVD-Median-Gated Streak Removal
///
/// 1. SVD(A) -> u, s, v (First component)
/// 2. v_smooth = MedianFilter(v)
/// 3. v_detail = v - v_smooth
/// 4. v_streak = Gate(v_detail, 3.0 * sigma)
/// 5. StreakImage = u * s * v_streak^T
/// 6. Corrected = A - StreakImage
pub fn svd_mg_removal(sinogram: ArrayView2<f64>) -> Array2<f64> {
    let (rows, cols) = sinogram.dim();

    // 1. Power Iteration (K=1)
    // 20 iterations is more than enough for streak dominance
    let (u, s, v) = power_iteration_k1(sinogram, 20, 1e-6);

    // 2. Filter v (Horizontal Profile)
    // Structure (Wide) vs Streak (Narrow)
    // Use Median Filter (51px window from research)
    let v_slice = v.as_slice().expect("Array must be standard layout");
    let v_smooth_vec = compute_1d_median_filter(v_slice, 51);
    let v_smooth = ndarray::Array1::from(v_smooth_vec);
    let v_detail = &v - &v_smooth;

    // 3. Magnitude Gating
    let sigma = estimate_robust_sigma(v_detail.view());
    let thresh = sigma * 3.0;

    // Gate function
    // mask = 1.0 / (1.0 + (abs(x)/thresh)^6)
    let v_streak = v_detail.mapv(|x| {
        let mask = 1.0 / (1.0 + (x.abs() / thresh).powi(6));
        x * mask
    });

    // 4. Reconstruct
    // Streak = s * (u * v_streak^T)
    let scaled_u = u.mapv(|x| x * s);

    // Outer product
    // Array2::from_shape_fn
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
