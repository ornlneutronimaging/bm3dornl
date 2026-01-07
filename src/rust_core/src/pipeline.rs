use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray2, ToPyArray};
use crate::block_matching::find_similar_patches;
use crate::transforms::{fft2d, ifft2d}; // We'll use inline 1D for now or fix transforms later
use rayon::prelude::*;
use ndarray::{Array2, Array3, ArrayView2, s};
use std::sync::atomic::{AtomicU32, Ordering};

/// Helper for atomic float addition
#[inline]
fn atomic_add_f32(dst: &AtomicU32, val: f32) {
    let mut current = dst.load(Ordering::Relaxed);
    loop {
        let current_f32 = f32::from_bits(current);
        let next_f32 = current_f32 + val;
        let next = next_f32.to_bits();
        match dst.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(x) => current = x,
        }
    }
}

/// Internal helper for BM3D Step Execution (Parallel)
/// This is not exposed to Python directly to ensure type safety and clarity.
fn run_bm3d_step(
    input_noisy: ArrayView2<f32>,
    input_pilot: ArrayView2<f32>,
    mode: usize, // 0: HT, 1: Wiener
    sigma_psd: ArrayView2<f32>, // PatchxPatch (colored static) or 1x1 (ignored if use_colored_noise=false effectively)
    sigma_map: ArrayView2<f32>, // SPATIALLY ADAPTIVE: (H, W) map of local correlated noise intensity (streaks)
    sigma_random: f32,          // Base Random Noise Standard Deviation
    threshold: f32,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> Array2<f32> {
    let (rows, cols) = input_noisy.dim();
    
    // Validate shapes
    if input_pilot.dim() != (rows, cols) {
        panic!("Noisy and Pilot images must have same dimensions");
    }
    // Validate sigma_map shape
    if sigma_map.dim() != (rows, cols) {
         // If 1x1 passed (no map), we can handle it, but better strict or check dim
         if sigma_map.dim() != (1, 1) {
             panic!("sigma_map must match image dimensions or be 1x1");
         }
    }
    let use_sigma_map = sigma_map.dim() == (rows, cols);
    
    // Check PSD shape
    let use_colored_noise = if sigma_psd.dim() == (patch_size, patch_size) {
        true
    } else if sigma_psd.dim() == (1, 1) {
        false
    } else {
         panic!("sigma_psd must be 1x1 (unused) or patch_size x patch_size");
    };
    
    let scalar_sigma_sq = sigma_random * sigma_random;

    // Accumulation buffers
    let num_elements = rows * cols;
    let mut numerator_vec: Vec<AtomicU32> = (0..num_elements).map(|_| AtomicU32::new(0)).collect();
    let mut denominator_vec: Vec<AtomicU32> = (0..num_elements).map(|_| AtomicU32::new(0)).collect();

    // Grid generation
    let mut ref_coords = Vec::new();
    let r_end = rows.saturating_sub(patch_size) + 1;
    let c_end = cols.saturating_sub(patch_size) + 1;
    
    for r in (0..r_end).step_by(step_size) {
        for c in (0..c_end).step_by(step_size) {
            ref_coords.push((r, c));
        }
    }
    
    let numerator_slice = &numerator_vec;
    let denominator_slice = &denominator_vec;
    
    // Use input_pilot for block matching
    let matching_target = input_pilot;

    ref_coords.par_iter().for_each(|&(ref_r, ref_c)| {
        // 1. Block Matching on PILOT
        let matches = find_similar_patches(
            matching_target, 
            (ref_r, ref_c), 
            (patch_size, patch_size), 
            (search_window, search_window), 
            max_matches,
            step_size 
        );
        let k = matches.len();
        if k == 0 { return; }
        
        // 1.5 Get Local Noise Level from Map (Using Reference Patch position)
        // We use the top-left value, or average? Top-left is fast.
        // Usually streaks are column-based, so checking `ref_c` is enough.
        // Assuming sigma_map is aligned.
        let local_sigma_streak = if use_sigma_map {
             sigma_map[[ref_r, ref_c]] // Takes the value at the corner. Could be improved to gather max/mean in patch.
        } else {
             0.0
        };

        let mut group_noisy = Array3::<f32>::zeros((k, patch_size, patch_size));
        let mut group_pilot = Array3::<f32>::zeros((k, patch_size, patch_size));
        
        for (idx, m) in matches.iter().enumerate() {
            let p_n = input_noisy.slice(s![m.row..m.row+patch_size, m.col..m.col+patch_size]);
            let p_p = input_pilot.slice(s![m.row..m.row+patch_size, m.col..m.col+patch_size]);
            group_noisy.slice_mut(s![idx, .., ..]).assign(&p_n);
            group_pilot.slice_mut(s![idx, .., ..]).assign(&p_p);
        }
        
        // Transform
        let mut g_noisy_c = ndarray::Array3::<rustfft::num_complex::Complex<f32>>::zeros((k, patch_size, patch_size));
        let mut g_pilot_c = ndarray::Array3::<rustfft::num_complex::Complex<f32>>::zeros((k, patch_size, patch_size));
        
        for i in 0..k {
             let slice_n = group_noisy.slice(s![i, .., ..]);
             let fft_n = fft2d(slice_n);
             g_noisy_c.slice_mut(s![i, .., ..]).assign(&fft_n);
             if mode == 1 {
                 let slice_p = group_pilot.slice(s![i, .., ..]);
                 let fft_p = fft2d(slice_p);
                 g_pilot_c.slice_mut(s![i, .., ..]).assign(&fft_p);
             }
        }
        
        // 1D FFT along K
        let mut planner = rustfft::FftPlanner::new();
        let fft_k = planner.plan_fft_forward(k);
        for r in 0..patch_size {
            for c in 0..patch_size {
                let mut vec_n: Vec<rustfft::num_complex::Complex<f32>> = (0..k).map(|i| g_noisy_c[[i, r, c]]).collect();
                fft_k.process(&mut vec_n);
                for i in 0..k { g_noisy_c[[i, r, c]] = vec_n[i]; }
                if mode == 1 {
                    let mut vec_p: Vec<rustfft::num_complex::Complex<f32>> = (0..k).map(|i| g_pilot_c[[i, r, c]]).collect();
                    fft_k.process(&mut vec_p);
                    for i in 0..k { g_pilot_c[[i, r, c]] = vec_p[i]; }
                }
            }
        }

        // 4. Filtering
        let mut weight_g = 1.0; 
        let spatial_scale = patch_size as f32;
        let spatial_scale_sq = (patch_size * patch_size) as f32;
        
        if mode == 0 {
            // Hard Thresholding
            let mut nz_count = 0;
            
             for i in 0..k {
                 // let is_dc = i == 0; // Unused
                 for r in 0..patch_size {
                     for c in 0..patch_size {
                         let coeff = g_noisy_c[[i, r, c]];
                         
                         let noise_std_coeff = {
                             let sigma_s_dist = if use_colored_noise { sigma_psd[[r, c]] } else { 0.0 };
                             // Spatially Adaptive: Scale the PSD shape by the Local Streak Intensity
                             let effective_sigma_s = sigma_s_dist * local_sigma_streak;
                             
                             // Variance Model:
                             // Var_Total = Var_Random + Var_Streak
                             // Var_Random = k * sigma_random^2 (White noise grows with sqrt(k) in transform? No, var grows with k)
                             // wait, Haar/Unitary transform preserves L2.
                             // Input Var = sigma^2.
                             // After 2D FFT (normalized): Var = sigma^2.
                             // After 1D FFT along K (unnormalized): Var = k * sigma^2.
                             
                             let var_r = k as f32 * scalar_sigma_sq;
                             let var_s = (k*k) as f32 * effective_sigma_s * effective_sigma_s; // Correlated noise grows with k^2
                             (var_r + var_s).sqrt() * spatial_scale
                         };
                         
                         let thresh_val = threshold * noise_std_coeff;
                         
                         if coeff.norm() < thresh_val {
                             g_noisy_c[[i, r, c]] = rustfft::num_complex::Complex::new(0.0, 0.0);
                         } else {
                             nz_count += 1;
                         }
                     }
                 }
             }

             if nz_count > 0 { weight_g = 1.0 / (nz_count as f32 + 1.0); } 
        } else {
             // Wiener Filtering
             let mut wiener_sum = 0.0;
             for i in 0..k {
                 // Apply aggressive sigma_psd logic to all i
                 for r in 0..patch_size {
                     for c in 0..patch_size {
                         let p_val = g_pilot_c[[i, r, c]];
                         let n_val = g_noisy_c[[i, r, c]];
                         let p_sq = p_val.norm_sqr();
                         
                         let noise_var_coeff = {
                              let sigma_s_dist = if use_colored_noise { sigma_psd[[r, c]] } else { 0.0 };
                              let effective_sigma_s = sigma_s_dist * local_sigma_streak;
                              
                              let var_r = k as f32 * scalar_sigma_sq;
                              let var_s = (k*k) as f32 * effective_sigma_s * effective_sigma_s;
                              (var_r + var_s) * spatial_scale_sq
                         };
                         
                         let w = p_sq / (p_sq + noise_var_coeff + 1e-8);
                         g_noisy_c[[i, r, c]] = n_val * w;
                         
                         wiener_sum += w*w; 
                     }
                 }
             }
             
             // Group weight: Simplification
             let noise_var_global = scalar_sigma_sq; 
             weight_g = 1.0 / (wiener_sum * noise_var_global + 1e-8); 
             if weight_g > 1e6 { weight_g = 1e6; }
        }

        // 5. Inverse transforms
        let ifft_k = planner.plan_fft_inverse(k);
        let norm_k = 1.0 / k as f32;
         for r in 0..patch_size {
            for c in 0..patch_size {
                let mut vec: Vec<rustfft::num_complex::Complex<f32>> = (0..k).map(|i| g_noisy_c[[i, r, c]]).collect();
                ifft_k.process(&mut vec);
                for i in 0..k { g_noisy_c[[i, r, c]] = vec[i] * norm_k; }
            }
        }
        
        for i in 0..k {
            let complex_slice = g_noisy_c.slice(s![i, .., ..]).to_owned();
            let spatial = ifft2d(&complex_slice); 
            
            let m = matches[i];
            
            let mut w_spatial = weight_g;
            for pr in 0..patch_size {
                for pc in 0..patch_size {
                    let val = spatial[[pr, pc]];
                    let target_idx = (m.row + pr) * cols + (m.col + pc);
                    if target_idx < num_elements {
                        atomic_add_f32(&numerator_slice[target_idx], val * w_spatial);
                        atomic_add_f32(&denominator_slice[target_idx], w_spatial);
                    }
                }
            }
        }
    });

    // Finalize
    let mut output = Array2::<f32>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            let num = f32::from_bits(numerator_vec[idx].load(Ordering::Relaxed));
            let den = f32::from_bits(denominator_vec[idx].load(Ordering::Relaxed));
            
            if den > 1e-6 {
                output[[r, c]] = num / den;
            } else {
                output[[r, c]] = input_noisy[[r, c]]; 
            }
        }
    }

    output
}

#[pyfunction]
pub fn bm3d_hard_thresholding<'py>(
    py: Python<'py>,
    input_noisy: PyReadonlyArray2<f32>,
    input_pilot: PyReadonlyArray2<f32>,
    sigma_psd: PyReadonlyArray2<f32>,
    sigma_map: PyReadonlyArray2<f32>, // New Arg
    sigma_random: f32, 
    threshold: f32,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> PyResult<&'py PyArray2<f32>> {
    let output = run_bm3d_step(
        input_noisy.as_array(),
        input_pilot.as_array(),
        0, // HT mode
        sigma_psd.as_array(),
        sigma_map.as_array(),
        sigma_random,
        threshold,
        patch_size,
        step_size,
        search_window,
        max_matches
    );
    Ok(output.to_pyarray(py))
}

#[pyfunction]
pub fn bm3d_wiener_filtering<'py>(
    py: Python<'py>,
    input_noisy: PyReadonlyArray2<f32>,
    input_pilot: PyReadonlyArray2<f32>,
    sigma_psd: PyReadonlyArray2<f32>,
    sigma_map: PyReadonlyArray2<f32>, // New Arg
    sigma_random: f32, 
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> PyResult<&'py PyArray2<f32>> {
    let output = run_bm3d_step(
        input_noisy.as_array(),
        input_pilot.as_array(),
        1, // Wiener mode
        sigma_psd.as_array(),
        sigma_map.as_array(),
        sigma_random,
        0.0, // threshold ignored
        patch_size,
        step_size,
        search_window,
        max_matches
    );
    Ok(output.to_pyarray(py))
}

#[pyfunction]
pub fn test_block_matching_rust(
    input: PyReadonlyArray2<f32>,
    ref_r: usize,
    ref_c: usize,
    patch_size: usize,
    search_win: usize,
    max_matches: usize,
) -> PyResult<Vec<(usize, usize, f32)>> {
    let input_arr = input.as_array();
    let matches = find_similar_patches(
        input_arr,
        (ref_r, ref_c),
        (patch_size, patch_size),
        (search_win, search_win),
        max_matches,
        1
    );
    let result = matches.into_iter().map(|m| (m.row, m.col, m.distance)).collect();
    Ok(result)
}
