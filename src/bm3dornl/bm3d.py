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
    mode: str = "generic",  # "generic" or "streak"
    sigma: float = 0.1,    
    block_matching_kwargs: dict = default_block_matching_kwargs,
    filter_kwargs: dict = default_filter_kwargs,
) -> np.ndarray:
    """Remove ring artifacts (streaks) from a sinogram or a stack of sinograms.
    
    Parameters
    ----------
    sinogram : np.ndarray
        The input data. Can be 2D (H, W) or 3D (N, H, W).
    mode : str
        "generic": Standard BM3D (assume white noise).
        "streak": Additive Streak Removal (Residual Median) + Standard BM3D.
    sigma : float
        Noise standard deviation.
        
    Returns
    -------
    np.ndarray
        Denoised data (same shape as input).
    """
    # Unpack parameters
    patch_size = block_matching_kwargs.get("patch_size", (7, 7))
    if isinstance(patch_size, int): patch_size_dim = patch_size
    else: patch_size_dim = patch_size[0]

    step_size = block_matching_kwargs.get("stride", 2)
    cut_off = block_matching_kwargs.get("cut_off_distance", (40, 40))[0]
    num_patches = block_matching_kwargs.get("num_patches_per_group", 64)
    threshold_hard = 2.7
    
    logger = logging.getLogger(__name__)
    
    # Check Dimension
    is_stack = sinogram.ndim == 3
    
    # Ensure float32 and normalize
    z = sinogram.astype(np.float32)
    d_min = z.min()
    d_max = z.max()
    z_norm = z
    if d_max > d_min:
        z_norm = (z - d_min) / (d_max - d_min)

    # --- Spatially Adaptive BM3D Setup ---
    # 1. Estimate Streak MAP (Spatially Varying Variance)
    sigma_map_arg = filter_kwargs.get("sigma_map", None)
    
    scale_factor = filter_kwargs.get("streak_sigma_scale", 1.1)
    
    # Helper to compute map for one 2D slice
    def compute_slice_map(slice_img):
        streak_profile_rough = estimate_streak_profile(slice_img, sigma_smooth=5.0, iterations=1)
        profile_smooth = gaussian_filter1d(streak_profile_rough, 20.0)
        streak_signal = streak_profile_rough - profile_smooth
        sigma_streak_1d = np.abs(streak_signal).astype(np.float32)
        return np.tile(sigma_streak_1d * scale_factor, (slice_img.shape[0], 1))

    if sigma_map_arg is None:
        if is_stack:
            # 3D: Compute per slice
            # We can loop. Pre-allocate.
            n, h, w = z_norm.shape
            sigma_map = np.zeros((n, h, w), dtype=np.float32)
            for i in range(n):
                sigma_map[i] = compute_slice_map(z_norm[i])
        else:
            # 2D
            sigma_map = compute_slice_map(z_norm)
    else:
        sigma_map = np.ascontiguousarray(sigma_map_arg, dtype=np.float32)
        if is_stack and sigma_map.ndim != 3:
            raise ValueError("sigma_map must be 3D for stack input")
    
    sigma_map = np.ascontiguousarray(sigma_map, dtype=np.float32)
    
    # 2. Sigma Random
    sigma_random = float(sigma)

    # --- Mode Support (Streak Subtraction) ---
    sigma_psd = np.zeros((1, 1), dtype=np.float32) 
    
    # Prepare PSD (Shared)
    if mode == "streak" and patch_size_dim > 0:
        sigma_psd = np.zeros((patch_size_dim, patch_size_dim), dtype=np.float32)
        y_coords = np.arange(patch_size_dim)
        psd_profile = np.exp(-0.5 * (y_coords / 0.6)**2) 
        for x in range(patch_size_dim):
            sigma_psd[:, x] = psd_profile
        sigma_psd = sigma_psd.astype(np.float32) 

    # Apply Pre-subtraction
    if mode == "streak":
        sigma_smooth = filter_kwargs.get("sigma_smooth", 3.0)
        iterations = filter_kwargs.get("streak_iterations", 2)
        
        if is_stack:
            n = z_norm.shape[0]
            for i in range(n):
                prof = estimate_streak_profile(z_norm[i], sigma_smooth=sigma_smooth, iterations=iterations)
                corr = np.tile(prof, (z_norm[i].shape[0], 1))
                z_norm[i] -= corr
        else:
            prof = estimate_streak_profile(z_norm, sigma_smooth=sigma_smooth, iterations=iterations)
            corr = np.tile(prof, (z_norm.shape[0], 1))
            z_norm -= corr

    # --- BM3D Execution ---
    
    if is_stack:
        # Stack Processing
        yhat_ht = bm3d_rust.bm3d_hard_thresholding_stack(
            z_norm, z_norm, sigma_psd, 
            sigma_map, sigma_random, 
            threshold_hard,
            patch_size_dim, step_size, cut_off, num_patches
        )
        
        yhat_final = bm3d_rust.bm3d_wiener_filtering_stack(
             z_norm, yhat_ht, sigma_psd, 
             sigma_map, sigma_random,
             patch_size_dim, step_size, cut_off, num_patches
        )
    else:
        # Single Slice
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
    
    # Denormalize and Return
    if d_max > d_min:
        yhat_final = yhat_final * (d_max - d_min) + d_min
    return yhat_final
