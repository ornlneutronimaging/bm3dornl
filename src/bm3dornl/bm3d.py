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
    """Remove ring artifacts (streaks) from a sinogram.
    
    Parameters
    ----------
    sinogram : np.ndarray
        The sinogram (2D float array).
    mode : str
        "generic": Standard BM3D (assume white noise).
        "streak": Additive Streak Removal (Residual Median) + Standard BM3D.
    sigma : float
        Noise standard deviation.
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
    
    # Ensure float32 and normalize
    z = sinogram.astype(np.float32)
    d_min = z.min()
    d_max = z.max()
    z_norm = z
    if d_max > d_min:
        z_norm = (z - d_min) / (d_max - d_min)

    # --- Spatially Adaptive BM3D Setup ---
    # 1. Estimate Streak MAP (Spatially Varying Variance)
    sigma_map = filter_kwargs.get("sigma_map", None)
    
    # If not provided, estimate from the single image (Data-Driven)
    if sigma_map is None:
        # Improved Data-Driven Estimation
        # We need a robust estimate of the LOCAL streak intensity.
        # Previous method: global residual median.
        # Better: Local standard deviation of the "streak component".
        
        # 1. Extract Streak Component roughly
        streak_profile_rough = estimate_streak_profile(z_norm, sigma_smooth=5.0, iterations=1)
        
        # 2. High-pass filter this profile? 
        # Actually, the streak profile itself contains the "magnitude" info.
        # But we want the "Noise Standard Deviation" map.
        # Where the streak is strong, the "correlated noise" variance is high.
        # We map |Streak| -> Sigma.
        
        # Remove low-freq trends from profile to get just the "spikes" (streaks)
        profile_smooth = gaussian_filter1d(streak_profile_rough, 20.0)
        streak_signal = streak_profile_rough - profile_smooth
        
        # Map is the Magnitude of the Streak Signal
        # We assume the "streak noise" variance is proportional to the streak intensity squared?
        # Or just linear? Linear is a good approximation for amplitude.
        sigma_streak_1d = np.abs(streak_signal).astype(np.float32)
        
        # Amplify?
        # User feedback suggests "slightly worse", maybe under-estimating.
        # Let's boost the estimate slightly for robust removal.
        scale_factor = filter_kwargs.get("streak_sigma_scale", 1.5)
        sigma_map = np.tile(sigma_streak_1d * scale_factor, (z_norm.shape[0], 1))
        
    sigma_map = np.ascontiguousarray(sigma_map, dtype=np.float32)
    
    # 2. Sigma Random (White Noise Floor)
    sigma_random = float(sigma)

    # --- Mode Support (Streak Subtraction) ---
    sigma_psd = np.zeros((1, 1), dtype=np.float32) 
    
    if mode == "streak":
        # Additional Mean Subtraction (Shift).
        # We keep this as it works well for the DC component.
        sigma_smooth = filter_kwargs.get("sigma_smooth", 5.0)
        iterations = filter_kwargs.get("streak_iterations", 2)
        
        # Iterative pre-subtraction
        # This removes the "mean" of the streak, leaving residual noise/wobbble
        # tailored for BM3D to clean up.
        current_streak_profile = estimate_streak_profile(z_norm, sigma_smooth=sigma_smooth, iterations=iterations)
        correction = np.tile(current_streak_profile, (z_norm.shape[0], 1))
        z_norm = z_norm - correction
        
        # Construct PSD for Vertical Streaks
        # Vertical streaks (constant in Y) corresponds to energy in the first row of the 2D transform (freq Y = 0)
        if patch_size_dim > 0:
            sigma_psd = np.zeros((patch_size_dim, patch_size_dim), dtype=np.float32)
            
            # Gaussian Profile along Y axis (Vertical Freq)
            # Centered at 0 (DC). Width determines how "wobbly" streaks can be.
            # Narrow width = perfectly vertical. Wider = varying.
            # We want to suppress low V-freqs.
            
            # Simple 1D Gaussian decay
            y_coords = np.arange(patch_size_dim)
            # Use small sigma for freq domain (concentration)
            psd_profile = np.exp(-0.5 * (y_coords / 0.8)**2) 
            
            # Replicate along X (all horizontal freqs affected equally for a vertical line)
            for x in range(patch_size_dim):
                sigma_psd[:, x] = psd_profile
                
            # Normalize peak to 1.0 (relative to sigma_map)
            sigma_psd = sigma_psd.astype(np.float32) 

    
    # --- BM3D Execution (Adaptive) ---
    
    # Step 1: Hard Thresholding
    # Note: We now pass sigma_map and sigma_random
    yhat_ht = bm3d_rust.bm3d_hard_thresholding(
        z_norm, z_norm, sigma_psd, 
        sigma_map, sigma_random, 
        threshold_hard,
        patch_size_dim, step_size, cut_off, num_patches
    )
    
    # Step 2: Wiener Filtering
    yhat_final = bm3d_rust.bm3d_wiener_filtering(
         z_norm, yhat_ht, sigma_psd, 
         sigma_map, sigma_random,
         patch_size_dim, step_size, cut_off, num_patches
    )
    
    # Denormalize and Return
    if d_max > d_min:
        yhat_final = yhat_final * (d_max - d_min) + d_min
    return yhat_final
