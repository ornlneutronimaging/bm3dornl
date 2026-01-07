use rustfft::{Fft, num_complex::Complex};
use ndarray::{Array2, Array3, ArrayView2};
use std::sync::Arc;

/// Compute 2D FFT of a square patch using pre-computed plans.
/// Returns unnormalized FFT.
pub fn fft2d(
    input: ArrayView2<f32>, 
    fft_row_plan: &Arc<dyn Fft<f32>>, 
    fft_col_plan: &Arc<dyn Fft<f32>>
) -> Array2<Complex<f32>> {
    let (rows, cols) = input.dim();
    
    // Validate plans
    // if fft_row_plan.len() != cols || fft_col_plan.len() != rows { panic!("FFT plan size mismatch"); }

    // 1. Transform rows
    // Optimization: We avoid creating a Vec<Complex> for every row if possible.
    // Ideally we would transpose, but for small patches (8x8, 16x16) overhead dominates.
    // We stick to row-by-row but use a scratch buffer? 
    // Since this is inside parallel execution, scratch buffer management is tricky.
    // Let's rely on the small allocation being fast or improve later.
    
    let mut intermediate = Array2::<Complex<f32>>::zeros((rows, cols));
    let mut row_vec = vec![Complex::new(0.0, 0.0); cols];

    for r in 0..rows {
        // Copy to buffer
        for (c, &v) in input.row(r).iter().enumerate() {
            row_vec[c] = Complex::new(v, 0.0);
        }
        // FFT
        fft_row_plan.process(&mut row_vec);
        // Copy back
        for c in 0..cols {
            intermediate[[r, c]] = row_vec[c];
        }
    }

    // 2. Transform columns
    let mut output = Array2::<Complex<f32>>::zeros((rows, cols));
    let mut col_vec = vec![Complex::new(0.0, 0.0); rows];

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
pub fn ifft2d(
    input: &Array2<Complex<f32>>,
    ifft_row_plan: &Arc<dyn Fft<f32>>,
    ifft_col_plan: &Arc<dyn Fft<f32>>
) -> Array2<f32> {
    let (rows, cols) = input.dim();

    // 1. Transform columns
    let mut intermediate = input.clone();
    let mut col_vec = vec![Complex::new(0.0, 0.0); rows];

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
    let mut output = Array2::<f32>::zeros((rows, cols));
    let norm_factor = 1.0 / (rows * cols) as f32;
    let mut row_vec = vec![Complex::new(0.0, 0.0); cols];
    
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

/// Compute 1D FFT along the first dimension (Group dimension).
/// Input shape: (K, P, P)
/// Output shape: (K, P, P) (Complex)
pub fn fft1d_group(
    input: &Array3<f32>,
    fft_k_plan: &Arc<dyn Fft<f32>>
) -> Array3<Complex<f32>> {
    let (k, rows, cols) = input.dim();
    // if fft_k_plan.len() != k { panic!("FFT plan K mismatch"); }

    let mut output = Array3::<Complex<f32>>::zeros((k, rows, cols));
    let mut vec = vec![Complex::new(0.0, 0.0); k];

    // Iterate over each pixel (r, c) and transform the vector along K
    for r in 0..rows {
        for c in 0..cols {
            for i in 0..k {
                vec[i] = Complex::new(input[[i, r, c]], 0.0);
            }
            fft_k_plan.process(&mut vec);
            for i in 0..k {
                output[[i, r, c]] = vec[i];
            }
        }
    }
    output
}

/// Compute 1D Inverse FFT along the first dimension.
pub fn ifft1d_group(
    input: &Array3<Complex<f32>>,
    ifft_k_plan: &Arc<dyn Fft<f32>>
) -> Array3<f32> {
    let (k, rows, cols) = input.dim();
    
    let mut output = Array3::<f32>::zeros((k, rows, cols));
    let norm = 1.0 / k as f32;
    let mut vec = vec![Complex::new(0.0, 0.0); k];

    for r in 0..rows {
        for c in 0..cols {
            for i in 0..k {
                vec[i] = input[[i, r, c]];
            }
            ifft_k_plan.process(&mut vec);
            for i in 0..k {
                output[[i, r, c]] = vec[i].re * norm;
            }
        }
    }
    output
}



/// In-place Fast Walsh-Hadamard Transform (Natural Order) for 8 elements
#[inline(always)]
fn fwht8(buf: &mut [f32; 8]) {
    // Stage 1 (Stride 1)
    let t0 = buf[0] + buf[1]; buf[1] = buf[0] - buf[1]; buf[0] = t0;
    let t2 = buf[2] + buf[3]; buf[3] = buf[2] - buf[3]; buf[2] = t2;
    let t4 = buf[4] + buf[5]; buf[5] = buf[4] - buf[5]; buf[4] = t4;
    let t6 = buf[6] + buf[7]; buf[7] = buf[6] - buf[7]; buf[6] = t6;

    // Stage 2 (Stride 2)
    let t0 = buf[0] + buf[2]; buf[2] = buf[0] - buf[2]; buf[0] = t0;
    let t1 = buf[1] + buf[3]; buf[3] = buf[1] - buf[3]; buf[1] = t1;
    let t4 = buf[4] + buf[6]; buf[6] = buf[4] - buf[6]; buf[4] = t4;
    let t5 = buf[5] + buf[7]; buf[7] = buf[5] - buf[7]; buf[5] = t5;

    // Stage 3 (Stride 4)
    let t0 = buf[0] + buf[4]; buf[4] = buf[0] - buf[4]; buf[0] = t0;
    let t1 = buf[1] + buf[5]; buf[5] = buf[1] - buf[5]; buf[1] = t1;
    let t2 = buf[2] + buf[6]; buf[6] = buf[2] - buf[6]; buf[2] = t2;
    let t3 = buf[3] + buf[7]; buf[7] = buf[3] - buf[7]; buf[3] = t3;
}

/// 2D WHT for 8x8 patch. Returns Complex (im=0) for compatibility.
pub fn wht2d_8x8_forward(input: ArrayView2<f32>) -> Array2<Complex<f32>> {
    let mut data = [0.0; 64];
    let mut idx = 0;
    for r in 0..8 {
        for c in 0..8 {
            data[idx] = input[[r, c]];
            idx += 1;
        }
    }
    for r in 0..8 {
        let mut row_buf = [0.0; 8];
        let offset = r * 8;
        row_buf.copy_from_slice(&data[offset..offset+8]);
        fwht8(&mut row_buf);
        data[offset..offset+8].copy_from_slice(&row_buf);
    }
    for c in 0..8 {
        let mut col_buf = [0.0; 8];
        for r in 0..8 {
            col_buf[r] = data[r*8 + c];
        }
        fwht8(&mut col_buf);
        for r in 0..8 {
            data[r*8 + c] = col_buf[r];
        }
    }
    let mut output = Array2::<Complex<f32>>::zeros((8, 8));
    idx = 0;
    for r in 0..8 {
        for c in 0..8 {
            output[[r, c]] = Complex::new(data[idx], 0.0);
            idx += 1;
        }
    }
    output
}

/// 2D Inverse WHT for 8x8 patch.
pub fn wht2d_8x8_inverse(input: &Array2<Complex<f32>>) -> Array2<f32> {
    let mut data = [0.0; 64];
    let mut idx = 0;
    for r in 0..8 {
        for c in 0..8 {
            data[idx] = input[[r, c]].re;
            idx += 1;
        }
    }
    for r in 0..8 {
        let mut row_buf = [0.0; 8];
        let offset = r * 8;
        row_buf.copy_from_slice(&data[offset..offset+8]);
        fwht8(&mut row_buf);
        data[offset..offset+8].copy_from_slice(&row_buf);
    }
    for c in 0..8 {
        let mut col_buf = [0.0; 8];
        for r in 0..8 {
            col_buf[r] = data[r*8 + c];
        }
        fwht8(&mut col_buf);
        for r in 0..8 {
            data[r*8 + c] = col_buf[r];
        }
    }
    let scale = 1.0 / 64.0;
    let mut output = Array2::<f32>::zeros((8, 8));
    idx = 0;
    for r in 0..8 {
        for c in 0..8 {
            output[[r, c]] = data[idx] * scale;
            idx += 1;
        }
    }
    output
}
