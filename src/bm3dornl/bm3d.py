#!/usr/bin/env python3
"""Denoising functions using Rust backend."""

import logging
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from . import bm3d_rust

# Default parameters (simplified for Rust backend)
default_block_matching_kwargs = {
    "patch_size": (7, 7),
    "stride": 2,
    "cut_off_distance": (40, 40),
    "num_patches_per_group": 64,
}
# Legacy filter kwargs preserved just for signature compatibility if needed
default_filter_kwargs = {}

def estimate_streak_profile(sinogram, sigma_smooth=3.0, iterations=3):
    """
    Estimate the static vertical streak profile using an iterative robust approach.
    
    Iteration helps to separate object structure from the static streaks more cleanly,
    especially when object features align vertically.
    """
    z_clean = sinogram.copy()
    streak_acc = np.zeros(sinogram.shape[1], dtype=np.float32)
    
    for _ in range(iterations):
        # 1. Smooth along columns to estimate object (Low Frequency in Y)
        # We use a large sigma for Y to blur out the streaks, small for X to keep edges
        # But for streaks (constant X), we mainly want to reject them.
        # Smoothing isotropicly or anisotropicly?
        # A strong vertical smooth will keep vertical structures (streaks).
        # We want to REMOVE streaks to get object.
        # Actually, Gaussian filter is not robust. Median filter is better for rejecting outliers (streaks).
        # But median is slow.
        
        # Standard approach: Smooth heavily to get object "trend"
        # Since streaks are high-freq in X (edges) and DC in Y.
        z_smooth = gaussian_filter(z_clean, (sigma_smooth, 3.0)) # Strong smooth vertical, moderate horizontal
        
        # 2. Residual
        residual = z_clean - z_smooth
        
        # 3. Estimate Streak update (Robust average along columns)
        streak_update = np.median(residual, axis=0) # (W,)
        
        # 4. Refine update (smooth slightly to remove single-pixel noise, keep streak structures)
        # Streaks can be sharp, so small sigma.
        streak_update = gaussian_filter1d(streak_update, 1.0)
        
        # Accumulate
        streak_acc += streak_update
        
        # Remove from current estimate for next iteration (cleaning the object)
        correction = np.tile(streak_update, (sinogram.shape[0], 1))
        z_clean = z_clean - correction
        
    return streak_acc

def bm3d_ring_artifact_removal(
    sinogram: np.ndarray,
    mode: str = "generic",
    sigma: float = 0.1,    
    block_matching_kwargs: dict = default_block_matching_kwargs,
    filter_kwargs: dict = default_filter_kwargs,
) -> np.ndarray:
    """Remove ring artifacts (streaks) from a sinogram or a stack of sinograms.

    This function leverages a customized BM3D implementation optimized for streak removal.
    It supports both 2D (single sinogram) and 3D (stack of sinograms) inputs.
    
    Parameters
    ----------
    sinogram : np.ndarray
        The input data. Can be 2D (H, W) or 3D (N, H, W).
    mode : str, optional
        Operation mode:
        - "generic": Standard BM3D (assume white noise).
        - "streak": Additive Streak Removal (Residual Median) + Standard BM3D.
        By default "generic".
    sigma : float, optional
        Noise standard deviation, by default 0.1.
    block_matching_kwargs : dict, optional
        Parameters for block matching:
        - patch_size (tuple[int, int] | int): Size of patches (e.g. (8,8) or 8).
        - stride (int): Step size for patch extraction.
        - cut_off_distance (tuple[int, int]): Max distance for block matching.
        - num_patches_per_group (int): Number of similar patches to group.
        - batch_size (int): Chunk size for stack processing (default 32).
        By default `default_block_matching_kwargs`.
    filter_kwargs : dict, optional
        Parameters for filtering:
        - sigma_map (np.ndarray): Optional pre-computed sigma map.
        - streak_sigma_scale (float): Scale factor for streak sigma estimation.
        - sigma_smooth (float): Sigma for smoothing in streak estimation.
        - streak_iterations (int): Iterations for streak estimation.
        By default `default_filter_kwargs`.
        
    Returns
    -------
    np.ndarray
        Denoised data with same shape as input.
    """
    # Unpack parameters
    patch_size = block_matching_kwargs.get("patch_size", (7, 7))
    if isinstance(patch_size, int): patch_size_dim = patch_size
    else: patch_size_dim = patch_size[0]

    step_size = block_matching_kwargs.get("stride", 2)
    cut_off = block_matching_kwargs.get("cut_off_distance", (40, 40))[0]
    num_patches = block_matching_kwargs.get("num_patches_per_group", 64)
    batch_size = block_matching_kwargs.get("batch_size", 32)
    threshold_hard = 2.7
    
    logger = logging.getLogger(__name__)
    
    # Check Dimension
    is_stack = sinogram.ndim == 3
    
    # 1. Global Stats (Min/Max)
    # We compute these globally to ensure consistent scaling across chunks.
    # Note: If memory is extremely tight, streaming min/max could be implemented, 
    # but sinogram.min() usually runs efficiently.
    d_min = float(sinogram.min())
    d_max = float(sinogram.max())

    # --- Spatially Adaptive BM3D Setup Helpers ---
    sigma_map_arg = filter_kwargs.get("sigma_map", None)
    scale_factor = filter_kwargs.get("streak_sigma_scale", 1.1)

    def compute_slice_map(slice_img):
        streak_profile_rough = estimate_streak_profile(slice_img, sigma_smooth=5.0, iterations=1)
        profile_smooth = gaussian_filter1d(streak_profile_rough, 20.0)
        streak_signal = streak_profile_rough - profile_smooth
        sigma_streak_1d = np.abs(streak_signal).astype(np.float32)
        return np.tile(sigma_streak_1d * scale_factor, (slice_img.shape[0], 1))

    # Prepare PSD (Shared)
    sigma_psd = np.zeros((1, 1), dtype=np.float32) 
    if mode == "streak" and patch_size_dim > 0:
        sigma_psd = np.zeros((patch_size_dim, patch_size_dim), dtype=np.float32)
        y_coords = np.arange(patch_size_dim)
        psd_profile = np.exp(-0.5 * (y_coords / 0.6)**2) 
        for x in range(patch_size_dim):
            sigma_psd[:, x] = psd_profile
        sigma_psd = sigma_psd.astype(np.float32) 

    sigma_random = float(sigma)
    sigma_smooth = filter_kwargs.get("sigma_smooth", 3.0)
    streak_iterations = filter_kwargs.get("streak_iterations", 2)

    # Output allocation
    # We allocate the output array upfront.
    output_result = np.empty_like(sinogram) 

    if is_stack:
        n_slices = sinogram.shape[0]
        # Validate sigma_map if provided
        sigma_map_full = None
        if sigma_map_arg is not None:
            sigma_map_full = np.ascontiguousarray(sigma_map_arg, dtype=np.float32)
            if sigma_map_full.ndim != 3:
                raise ValueError("sigma_map must be 3D for stack input")

        # Process in batches to control memory usage
        for i in range(0, n_slices, batch_size):
            end = min(i + batch_size, n_slices)
            
            # 1. Extract and Normalize Chunk
            # Use astype to ensure float32 and create a copy for processing
            chunk = sinogram[i:end].astype(np.float32) 
            
            z_chunk = chunk
            # Normalize
            if d_max > d_min:
                z_chunk = (chunk - d_min) / (d_max - d_min)
            
            # 2. Sigma Map Chunk
            if sigma_map_full is not None:
                chunk_map = sigma_map_full[i:end]
            else:
                # Compute per slice
                c_n, c_h, c_w = z_chunk.shape
                chunk_map = np.zeros((c_n, c_h, c_w), dtype=np.float32)
                for k in range(c_n):
                    chunk_map[k] = compute_slice_map(z_chunk[k])
            
            chunk_map = np.ascontiguousarray(chunk_map, dtype=np.float32)

            # 3. Streak Pre-Subtraction
            if mode == "streak":
                # Operate on z_chunk in-place
                for k in range(z_chunk.shape[0]):
                    prof = estimate_streak_profile(z_chunk[k], sigma_smooth=sigma_smooth, iterations=streak_iterations)
                    corr = np.tile(prof, (z_chunk[k].shape[0], 1))
                    z_chunk[k] -= corr

            # 4. Run Rust Stack Processing
            # Note: Rust takes &Array3, returns new Array3.
            yhat_ht = bm3d_rust.bm3d_hard_thresholding_stack(
                z_chunk, z_chunk, sigma_psd, 
                chunk_map, sigma_random, 
                threshold_hard,
                patch_size_dim, step_size, cut_off, num_patches
            )
            
            yhat_final_chunk = bm3d_rust.bm3d_wiener_filtering_stack(
                z_chunk, yhat_ht, sigma_psd, 
                chunk_map, sigma_random,
                patch_size_dim, step_size, cut_off, num_patches
            )
            
            # 5. Denormalize and Store
            if d_max > d_min:
                yhat_final_chunk = yhat_final_chunk * (d_max - d_min) + d_min
            
            output_result[i:end] = yhat_final_chunk
            
            # Clean up explicit large temps if Python GC is lazy (optional)
            del z_chunk
            del chunk_map
            del yhat_ht
            del yhat_final_chunk

    else:
        # 2D Case (Single Slice)
        z = sinogram.astype(np.float32)
        z_norm = z
        if d_max > d_min:
            z_norm = (z - d_min) / (d_max - d_min)

        if sigma_map_arg is not None:
             sigma_map = np.ascontiguousarray(sigma_map_arg, dtype=np.float32)
        else:
             sigma_map = compute_slice_map(z_norm)
        
        sigma_map = np.ascontiguousarray(sigma_map, dtype=np.float32)

        if mode == "streak":
            prof = estimate_streak_profile(z_norm, sigma_smooth=sigma_smooth, iterations=streak_iterations)
            corr = np.tile(prof, (z_norm.shape[0], 1))
            z_norm -= corr

        yhat_ht = bm3d_rust.bm3d_hard_thresholding(
            z_norm, z_norm, sigma_psd, 
            sigma_map, sigma_random, 
            threshold_hard,
            patch_size_dim, step_size, cut_off, num_patches
        )
        
        yhat_final = bm3d_rust.bm3d_wiener_filtering(
            z_norm, yhat_ht, sigma_psd, 
            sigma_map, sigma_random,
            patch_size_dim, step_size, cut_off, num_patches
        )

        if d_max > d_min:
            yhat_final = yhat_final * (d_max - d_min) + d_min
        
        output_result = yhat_final

    return output_result
