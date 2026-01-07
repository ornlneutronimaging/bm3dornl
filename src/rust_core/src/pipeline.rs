use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray2, ToPyArray};
use crate::block_matching::find_similar_patches;
use rayon::prelude::*;
use ndarray::{Array2, Array3, ArrayView2, s};

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

    // --- FFT Plan Pre-computation ---
    let mut planner = rustfft::FftPlanner::new();
    // Only pre-compute what's needed for *this* step to avoid overhead?
    // Actually plans are cheap to create if sizes are small.
    let fft_2d_row = planner.plan_fft_forward(patch_size);
    let fft_2d_col = planner.plan_fft_forward(patch_size);
    let ifft_2d_row = planner.plan_fft_inverse(patch_size);
    let ifft_2d_col = planner.plan_fft_inverse(patch_size);
    
    let mut fft_1d_plans = Vec::with_capacity(max_matches + 1);
    let mut ifft_1d_plans = Vec::with_capacity(max_matches + 1);
    
    fft_1d_plans.push(planner.plan_fft_forward(1)); 
    ifft_1d_plans.push(planner.plan_fft_inverse(1));

    for k in 1..=max_matches {
        fft_1d_plans.push(planner.plan_fft_forward(k));
        ifft_1d_plans.push(planner.plan_fft_inverse(k));
    }
    
    // Grid generation
    let mut ref_coords = Vec::new();
    let r_end = rows.saturating_sub(patch_size) + 1;
    let c_end = cols.saturating_sub(patch_size) + 1;
    
    for r in (0..r_end).step_by(step_size) {
        for c in (0..c_end).step_by(step_size) {
            ref_coords.push((r, c));
        }
    }
    
    // Local refs for closure
    let fft_2d_row_ref = &fft_2d_row;
    let fft_2d_col_ref = &fft_2d_col;
    let ifft_2d_row_ref = &ifft_2d_row;
    let ifft_2d_col_ref = &ifft_2d_col;
    let fft_1d_plans_ref = &fft_1d_plans;
    let ifft_1d_plans_ref = &ifft_1d_plans;
    
    // Chunk size hint: give each thread decent work to amortize allocation
    // 512x512 with step 4 ~ 16k patches. 16 threads -> 1k patches per thread.
    // Allocation is 2x 512x512 floats ~ 2MB.
    // It's efficient.

    let (final_num, final_den) = ref_coords.par_iter()
        .with_min_len(2048)
        .fold(
            || (Array2::<f32>::zeros((rows, cols)), Array2::<f32>::zeros((rows, cols))),
            |mut acc, &(ref_r, ref_c)| {
                let (numerator_acc, denominator_acc) = &mut acc;

                // 1. Block Matching on PILOT
                let matches = crate::block_matching::find_similar_patches(
                    input_pilot, 
                    (ref_r, ref_c), 
                    (patch_size, patch_size), 
                    (search_window, search_window), 
                    max_matches,
                    step_size 
                );
                let k = matches.len();
                if k == 0 { return acc; }
                
                // 1.5 Get Local Noise Level
                let local_sigma_streak = if use_sigma_map {
                     sigma_map[[ref_r, ref_c]] 
                } else {
                     0.0
                };

                // Stack building
                let mut group_noisy = Array3::<f32>::zeros((k, patch_size, patch_size));
                let mut group_pilot = Array3::<f32>::zeros((k, patch_size, patch_size));
                
                for (idx, m) in matches.iter().enumerate() {
                    let p_n = input_noisy.slice(s![m.row..m.row+patch_size, m.col..m.col+patch_size]);
                    let p_p = input_pilot.slice(s![m.row..m.row+patch_size, m.col..m.col+patch_size]);
                    group_noisy.slice_mut(s![idx, .., ..]).assign(&p_n);
                    group_pilot.slice_mut(s![idx, .., ..]).assign(&p_p);
                }
                
                // 2D Transform
                let mut g_noisy_c = ndarray::Array3::<rustfft::num_complex::Complex<f32>>::zeros((k, patch_size, patch_size));
                let mut g_pilot_c = ndarray::Array3::<rustfft::num_complex::Complex<f32>>::zeros((k, patch_size, patch_size));
                
                for i in 0..k {
                     let slice_n = group_noisy.slice(s![i, .., ..]);
                     let fft_n = crate::transforms::fft2d(slice_n, fft_2d_row_ref, fft_2d_col_ref);
                     g_noisy_c.slice_mut(s![i, .., ..]).assign(&fft_n);
                     if mode == 1 {
                         let slice_p = group_pilot.slice(s![i, .., ..]);
                         let fft_p = crate::transforms::fft2d(slice_p, fft_2d_row_ref, fft_2d_col_ref);
                         g_pilot_c.slice_mut(s![i, .., ..]).assign(&fft_p);
                     }
                }
                
                // 1D Transform along K
                let fft_k_plan = &fft_1d_plans_ref[k]; 
                
                for r in 0..patch_size {
                    for c in 0..patch_size {
                        let mut vec_n: Vec<rustfft::num_complex::Complex<f32>> = (0..k).map(|i| g_noisy_c[[i, r, c]]).collect();
                        fft_k_plan.process(&mut vec_n);
                        for i in 0..k { g_noisy_c[[i, r, c]] = vec_n[i]; }
                        
                        if mode == 1 {
                            let mut vec_p: Vec<rustfft::num_complex::Complex<f32>> = (0..k).map(|i| g_pilot_c[[i, r, c]]).collect();
                            fft_k_plan.process(&mut vec_p);
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
                         for r in 0..patch_size {
                             for c in 0..patch_size {
                                 let coeff = g_noisy_c[[i, r, c]];
                                 let noise_std_coeff = {
                                     let sigma_s_dist = if use_colored_noise { sigma_psd[[r, c]] } else { 0.0 };
                                     let effective_sigma_s = sigma_s_dist * local_sigma_streak;
                                     let var_r = k as f32 * scalar_sigma_sq;
                                     let var_s = (k*k) as f32 * effective_sigma_s * effective_sigma_s; 
                                     (var_r + var_s).sqrt() * spatial_scale
                                 };
                                 if coeff.norm() < threshold * noise_std_coeff {
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
                         for r in 0..patch_size {
                             for c in 0..patch_size {
                                 let p_val = g_pilot_c[[i, r, c]];
                                 let n_val = g_noisy_c[[i, r, c]];
                                 
                                 let noise_var_coeff = {
                                       let sigma_s_dist = if use_colored_noise { sigma_psd[[r, c]] } else { 0.0 };
                                       let effective_sigma_s = sigma_s_dist * local_sigma_streak;
                                       let var_r = k as f32 * scalar_sigma_sq;
                                       let var_s = (k*k) as f32 * effective_sigma_s * effective_sigma_s;
                                       (var_r + var_s) * spatial_scale_sq
                                 };
                                 let w = p_val.norm_sqr() / (p_val.norm_sqr() + noise_var_coeff + 1e-8);
                                 g_noisy_c[[i, r, c]] = n_val * w;
                                 wiener_sum += w*w; 
                             }
                         }
                     }
                     let noise_var_global = scalar_sigma_sq; 
                     weight_g = 1.0 / (wiener_sum * noise_var_global + 1e-8); 
                     if weight_g > 1e6 { weight_g = 1e6; }
                }

                // 5. Inverse transforms
                let ifft_k_plan = &ifft_1d_plans_ref[k];
                let norm_k = 1.0 / k as f32;
                 for r in 0..patch_size {
                    for c in 0..patch_size {
                        let mut vec: Vec<rustfft::num_complex::Complex<f32>> = (0..k).map(|i| g_noisy_c[[i, r, c]]).collect();
                        ifft_k_plan.process(&mut vec);
                        for i in 0..k { g_noisy_c[[i, r, c]] = vec[i] * norm_k; }
                    }
                }
                
                for i in 0..k {
                    let complex_slice = g_noisy_c.slice(s![i, .., ..]).to_owned();
                    let spatial = crate::transforms::ifft2d(&complex_slice, ifft_2d_row_ref, ifft_2d_col_ref); 
                    
                    let m = matches[i];
                    let w_spatial = weight_g; // Removed mut, plain copy is fine
                    
                    // Accumulate to Thread-Local Buffer (No Atomics!)
                    // Bounds check is implicit or we can be careful.
                    // patches should stay within image if loop bounds are correct.
                    // But slice is safe.
                    // We can slice the accumulator? Or just loop.
                    for pr in 0..patch_size {
                        for pc in 0..patch_size {
                            let tr = m.row + pr;
                            let tc = m.col + pc;
                            if tr < rows && tc < cols {
                                numerator_acc[[tr, tc]] += spatial[[pr, pc]] * w_spatial;
                                denominator_acc[[tr, tc]] += w_spatial;
                            }
                        }
                    }
                }
                
                acc
            }
        )
        .reduce(
            || (Array2::<f32>::zeros((rows, cols)), Array2::<f32>::zeros((rows, cols))),
            |mut a, b| {
                a.0 += &b.0;
                a.1 += &b.1;
                a
            }
        );

    // Finalize
    let mut output = Array2::<f32>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let num = final_num[[r, c]];
            let den = final_den[[r, c]];
            
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
