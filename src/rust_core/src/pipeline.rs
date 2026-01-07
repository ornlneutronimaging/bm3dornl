use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray2, PyReadonlyArray3, PyArray3, ToPyArray};
use crate::block_matching::find_similar_patches;
use rayon::prelude::*;
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, s, Axis};
use std::sync::Arc;
use rustfft::Fft;

struct Bm3dPlans {
    fft_2d_row: Arc<dyn Fft<f32>>,
    fft_2d_col: Arc<dyn Fft<f32>>,
    ifft_2d_row: Arc<dyn Fft<f32>>,
    ifft_2d_col: Arc<dyn Fft<f32>>,
    fft_1d_plans: Vec<Arc<dyn Fft<f32>>>,
    ifft_1d_plans: Vec<Arc<dyn Fft<f32>>>,
}

impl Bm3dPlans {
    fn new(patch_size: usize, max_matches: usize) -> Self {
        let mut planner = rustfft::FftPlanner::new();
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

        Self {
            fft_2d_row, fft_2d_col, ifft_2d_row, ifft_2d_col,
            fft_1d_plans, ifft_1d_plans
        }
    }
}

/// Core BM3D Single Image Kernel
fn run_bm3d_kernel(
    input_noisy: ArrayView2<f32>,
    input_pilot: ArrayView2<f32>,
    mode: usize,
    sigma_psd: ArrayView2<f32>,
    sigma_map: ArrayView2<f32>,
    sigma_random: f32,
    threshold: f32,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
    plans: &Bm3dPlans,
) -> Array2<f32> {
    let (rows, cols) = input_noisy.dim();
    let use_sigma_map = sigma_map.dim() == (rows, cols);
    let use_colored_noise = sigma_psd.dim() == (patch_size, patch_size);
    let scalar_sigma_sq = sigma_random * sigma_random;
    
    // Fast path for 8x8 patches using Hadamard
    let use_hadamard = patch_size == 8;
    
    // Pre-compute Integral Images for Block Matching acceleration
    let (integral_sum, integral_sq_sum) = crate::block_matching::compute_integral_images(input_pilot);

    let mut ref_coords = Vec::new();
    let r_end = rows.saturating_sub(patch_size) + 1;
    let c_end = cols.saturating_sub(patch_size) + 1;
    
    for r in (0..r_end).step_by(step_size) {
        for c in (0..c_end).step_by(step_size) {
            ref_coords.push((r, c));
        }
    }

    let fft_2d_row_ref = &plans.fft_2d_row;
    let fft_2d_col_ref = &plans.fft_2d_col;
    let ifft_2d_row_ref = &plans.ifft_2d_row;
    let ifft_2d_col_ref = &plans.ifft_2d_col;
    let fft_1d_plans_ref = &plans.fft_1d_plans;
    let ifft_1d_plans_ref = &plans.ifft_1d_plans;

    let (final_num, final_den) = ref_coords.par_iter()
        .with_min_len(2048)
        .fold(
            || (Array2::<f32>::zeros((rows, cols)), Array2::<f32>::zeros((rows, cols))),
            |mut acc, &(ref_r, ref_c)| {
                let (numerator_acc, denominator_acc) = &mut acc;

                // 1. Block Matching
                let matches = crate::block_matching::find_similar_patches(
                    input_pilot, 
                    &integral_sum,
                    &integral_sq_sum,
                    (ref_r, ref_c), 
                    (patch_size, patch_size), 
                    (search_window, search_window), 
                    max_matches,
                    step_size 
                );
                let k = matches.len();
                if k == 0 { return acc; }
                
                // 1.5 Local Noise Level
                let local_sigma_streak = if use_sigma_map { sigma_map[[ref_r, ref_c]] } else { 0.0 };

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
                     if use_hadamard {
                         let fft_n = crate::transforms::wht2d_8x8_forward(slice_n);
                         g_noisy_c.slice_mut(s![i, .., ..]).assign(&fft_n);
                     } else {
                         let fft_n = crate::transforms::fft2d(slice_n, fft_2d_row_ref, fft_2d_col_ref);
                         g_noisy_c.slice_mut(s![i, .., ..]).assign(&fft_n);
                     }
                     
                     if mode == 1 {
                         let slice_p = group_pilot.slice(s![i, .., ..]);
                         if use_hadamard {
                             let fft_p = crate::transforms::wht2d_8x8_forward(slice_p);
                             g_pilot_c.slice_mut(s![i, .., ..]).assign(&fft_p);
                         } else {
                             let fft_p = crate::transforms::fft2d(slice_p, fft_2d_row_ref, fft_2d_col_ref);
                             g_pilot_c.slice_mut(s![i, .., ..]).assign(&fft_p);
                         }
                     }
                }
                
                // 1D Transform
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

                // Filtering
                let mut weight_g = 1.0; 
                let spatial_scale = patch_size as f32;
                let spatial_scale_sq = spatial_scale * spatial_scale;
                
                if mode == 0 {
                     // HT
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
                     // Wiener
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
                     weight_g = 1.0 / (wiener_sum * scalar_sigma_sq + 1e-8); 
                     if weight_g > 1e6 { weight_g = 1e6; }
                }

                // Inverse Transforms
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
                    let spatial = if use_hadamard {
                        crate::transforms::wht2d_8x8_inverse(&complex_slice)
                    } else {
                        crate::transforms::ifft2d(&complex_slice, ifft_2d_row_ref, ifft_2d_col_ref)
                    };
                    
                    let m = matches[i];
                    let w_spatial = weight_g;
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
            })
        .reduce(
            || (Array2::<f32>::zeros((rows, cols)), Array2::<f32>::zeros((rows, cols))),
            |mut a, b| { a.0 += &b.0; a.1 += &b.1; a }
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


fn run_bm3d_step(
    input_noisy: ArrayView2<f32>,
    input_pilot: ArrayView2<f32>,
    mode: usize, 
    sigma_psd: ArrayView2<f32>, 
    sigma_map: ArrayView2<f32>, 
    sigma_random: f32,          
    threshold: f32,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> Array2<f32> {
    if input_pilot.dim() != input_noisy.dim() { panic!("Dim mismatch"); }
    if sigma_map.dim() != input_noisy.dim() && sigma_map.dim() != (1, 1) { panic!("Map dim mismatch"); }
    
    let plans = Bm3dPlans::new(patch_size, max_matches);
    run_bm3d_kernel(input_noisy, input_pilot, mode, sigma_psd, sigma_map, sigma_random, threshold, patch_size, step_size, search_window, max_matches, &plans)
}

fn run_bm3d_step_stack(
    input_noisy: ArrayView3<f32>,
    input_pilot: ArrayView3<f32>,
    mode: usize, 
    sigma_psd: ArrayView2<f32>, 
    sigma_map: ArrayView3<f32>, 
    sigma_random: f32,          
    threshold: f32,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> Array3<f32> {
    let (n, rows, cols) = input_noisy.dim();
    if input_pilot.dim() != (n, rows, cols) { panic!("Stack dim mismatch pilot"); }
    if sigma_map.dim() != (n, rows, cols) && sigma_map.dim() != (1, 1, 1) { 
        // Support broadcasting if map is 1x1x1?
        if sigma_map.dim() == (1,1,1) {
            // we handle it inside loop effectively? No, slice fails.
            // We should expect full map or broadcast carefully.
            // To simplify, we demand full map if used, or dummy 1x1x1 is handled via logic.
            // But strict check is safer.
        } else {
            panic!("Stack dim mismatch map");
        }
    }
    
    let plans = Bm3dPlans::new(patch_size, max_matches);
    
    let results: Vec<Array2<f32>> = (0..n).into_par_iter()
        .map(|i| {
            let noisy_slice = input_noisy.index_axis(Axis(0), i);
            let pilot_slice = input_pilot.index_axis(Axis(0), i);
            let map_slice = if sigma_map.dim() == (1,1,1) {
                 sigma_map.index_axis(Axis(0), 0) // Dummy view
            } else {
                 sigma_map.index_axis(Axis(0), i)
            };
            
            run_bm3d_kernel(
                noisy_slice, pilot_slice, mode, sigma_psd, map_slice,
                sigma_random, threshold, patch_size, step_size, search_window, max_matches, &plans
            )
        })
        .collect();
        
    // Consolidate
    let mut output = Array3::<f32>::zeros((n, rows, cols));
    for (i, res) in results.into_iter().enumerate() {
        output.slice_mut(s![i, .., ..]).assign(&res);
    }
    output
}


#[pyfunction]
pub fn bm3d_hard_thresholding<'py>(
    py: Python<'py>,
    input_noisy: PyReadonlyArray2<f32>,
    input_pilot: PyReadonlyArray2<f32>,
    sigma_psd: PyReadonlyArray2<f32>,
    sigma_map: PyReadonlyArray2<f32>, 
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
        0, 
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
    sigma_map: PyReadonlyArray2<f32>,
    sigma_random: f32, 
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> PyResult<&'py PyArray2<f32>> {
    let output = run_bm3d_step(
        input_noisy.as_array(),
        input_pilot.as_array(),
        1, 
        sigma_psd.as_array(),
        sigma_map.as_array(),
        sigma_random,
        0.0,
        patch_size,
        step_size,
        search_window,
        max_matches
    );
    Ok(output.to_pyarray(py))
}

#[pyfunction]
pub fn bm3d_hard_thresholding_stack<'py>(
    py: Python<'py>,
    input_noisy: PyReadonlyArray3<f32>,
    input_pilot: PyReadonlyArray3<f32>,
    sigma_psd: PyReadonlyArray2<f32>,
    sigma_map: PyReadonlyArray3<f32>, 
    sigma_random: f32, 
    threshold: f32,
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> PyResult<&'py PyArray3<f32>> {
    let output = run_bm3d_step_stack(
        input_noisy.as_array(),
        input_pilot.as_array(),
        0, 
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
pub fn bm3d_wiener_filtering_stack<'py>(
    py: Python<'py>,
    input_noisy: PyReadonlyArray3<f32>,
    input_pilot: PyReadonlyArray3<f32>,
    sigma_psd: PyReadonlyArray2<f32>,
    sigma_map: PyReadonlyArray3<f32>,
    sigma_random: f32, 
    patch_size: usize,
    step_size: usize,
    search_window: usize,
    max_matches: usize,
) -> PyResult<&'py PyArray3<f32>> {
    let output = run_bm3d_step_stack(
        input_noisy.as_array(),
        input_pilot.as_array(),
        1, 
        sigma_psd.as_array(),
        sigma_map.as_array(),
        sigma_random,
        0.0,
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
    let (sum_img, sq_sum_img) = crate::block_matching::compute_integral_images(input_arr);
    let matches = crate::block_matching::find_similar_patches(
        input_arr,
        &sum_img,
        &sq_sum_img,
        (ref_r, ref_c),
        (patch_size, patch_size),
        (search_win, search_win),
        max_matches,
        1
    );
    let result = matches.into_iter().map(|m| (m.row, m.col, m.distance)).collect();
    Ok(result)
}
