use rustfft::{FftPlanner, num_complex::Complex};
use ndarray::{Array2, Array3, ArrayView2, Axis};
use std::sync::Arc;

/// Compute 2D FFT of a square patch.
/// Returns unnormalized FFT.
pub fn fft2d(input: ArrayView2<f32>) -> Array2<Complex<f32>> {
    let (rows, cols) = input.dim();
    let mut planner = FftPlanner::new();
    let fft_row = planner.plan_fft_forward(cols);
    let fft_col = planner.plan_fft_forward(rows);

    // 1. Transform rows
    let mut intermediate = Array2::<Complex<f32>>::zeros((rows, cols));
    for r in 0..rows {
        let mut row_vec: Vec<Complex<f32>> = input.row(r).iter().map(|&v| Complex::new(v, 0.0)).collect();
        fft_row.process(&mut row_vec);
        for c in 0..cols {
            intermediate[[r, c]] = row_vec[c];
        }
    }

    // 2. Transform columns
    let mut output = Array2::<Complex<f32>>::zeros((rows, cols));
    for c in 0..cols {
        let mut col_vec: Vec<Complex<f32>> = intermediate.column(c).to_vec();
        fft_col.process(&mut col_vec);
        for r in 0..rows {
            output[[r, c]] = col_vec[r];
        }
    }

    output
}

/// Compute 2D Inverse FFT of a square patch.
/// Normalizes by 1/(rows*cols).
pub fn ifft2d(input: &Array2<Complex<f32>>) -> Array2<f32> {
    let (rows, cols) = input.dim();
    let mut planner = FftPlanner::new();
    let fft_row = planner.plan_fft_inverse(cols);
    let fft_col = planner.plan_fft_inverse(rows);

    // 1. Transform columns
    let mut intermediate = input.clone();
    for c in 0..cols {
        let mut col_vec: Vec<Complex<f32>> = intermediate.column(c).to_vec();
        fft_col.process(&mut col_vec);
        for r in 0..rows {
            intermediate[[r, c]] = col_vec[r];
        }
    }

    // 2. Transform rows
    let mut output = Array2::<f32>::zeros((rows, cols));
    let norm_factor = 1.0 / (rows * cols) as f32;
    
    for r in 0..rows {
        let mut row_vec: Vec<Complex<f32>> = intermediate.row(r).to_vec();
        fft_row.process(&mut row_vec);
        for c in 0..cols {
            // Take real part (input was real)
            output[[r, c]] = row_vec[c].re * norm_factor;
        }
    }

    output
}

/// Compute 1D FFT along the first dimension (Group dimension).
/// Input shape: (K, P, P)
/// Output shape: (K, P, P) (Complex)
pub fn fft1d_group(input: &Array3<f32>) -> Array3<Complex<f32>> {
    let (k, rows, cols) = input.dim();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(k);

    let mut output = Array3::<Complex<f32>>::zeros((k, rows, cols));

    // Iterate over each pixel (r, c) and transform the vector along K
    for r in 0..rows {
        for c in 0..cols {
            let mut vec: Vec<Complex<f32>> = (0..k).map(|i| Complex::new(input[[i, r, c]], 0.0)).collect();
            fft.process(&mut vec);
            for i in 0..k {
                output[[i, r, c]] = vec[i];
            }
        }
    }
    output
}

/// Compute 1D Inverse FFT along the first dimension.
pub fn ifft1d_group(input: &Array3<Complex<f32>>) -> Array3<f32> {
    let (k, rows, cols) = input.dim();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(k);

    let mut output = Array3::<f32>::zeros((k, rows, cols));
    let norm = 1.0 / k as f32;

    for r in 0..rows {
        for c in 0..cols {
            let mut vec: Vec<Complex<f32>> = (0..k).map(|i| input[[i, r, c]]).collect();
            fft.process(&mut vec);
            for i in 0..k {
                output[[i, r, c]] = vec[i].re * norm;
            }
        }
    }
    output
}

