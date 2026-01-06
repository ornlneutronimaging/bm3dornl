#!/usr/bin/env python3
"""Denoising functions using Rust backend."""

import logging
import numpy as np
from skimage.transform import pyramid_laplacian, pyramid_expand
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

from scipy.ndimage import zoom

from scipy.ndimage import gaussian_filter1d

from scipy.signal import convolve
from scipy.signal.windows import gaussian
from scipy.fft import fft2

def estimate_streak_sigma(sinogram: np.ndarray, patch_size: tuple[int, int]) -> float:
    """Estimate the standard deviation of the streak noise using robust statistics on vertical profile.
    
    1. Compute mean vertical profile (averaging out random noise).
    2. High-pass filter (subtract smooth trend) to isolate streaks from object structure.
    3. Use MAD (Median Absolute Deviation) to estimate noise level, ignoring outliers (edges).
    """
    profile = np.mean(sinogram, axis=0) # Shape (W,)
    
    # 2. Separate Streaks from Object
    sigma_smooth = 15.0 # Tunable parameter for object smoothness
    profile_smooth = gaussian_filter1d(profile, sigma_smooth)
    
    streak_signal = profile - profile_smooth
    
    # 3. Robust Estimation (MAD)
    median_val = np.median(streak_signal)
    mad = np.median(np.abs(streak_signal - median_val))
    
    sigma_est = mad * 1.4826
    
    return sigma_est

def construct_streak_psd(
    patch_size: tuple[int, int], 
    sigma: float, 
    sinogram: np.ndarray,
    streak_strength: float = 1.0
) -> np.ndarray:
    """Construct colored noise PSD using robust MAD estimation."""
    H, W = patch_size
    # Base Random Noise PSD (White) -> Handled by separate scalar in Rust.
    # This map should ONLY contain the STATIC noise component.
    sigma_map = np.zeros((H, W), dtype=np.float32)
    
    # 1. Estimate robust streak variance
    sigma_streak = estimate_streak_sigma(sinogram, patch_size)
    
    # 2. Scale by strength
    sigma_target = sigma_streak * streak_strength
    
    # 3. Populate DC Row
    sigma_map[0, :] = sigma_target
    
    # PROTECT DC Component (Patch Mean / Global Brightness)
    sigma_map[0, 0] = sigma
    
    return sigma_map

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
        "streak": Multiscale BM3D with Streak PSD (remove vertical lines).
    sigma : float
        Noise standard deviation.
    """
    # Unpack parameters
    patch_size = block_matching_kwargs.get("patch_size", (7, 7))
    if isinstance(patch_size, int): patch_size_dim = patch_size
    else: patch_size_dim = patch_size[0] # Assume square for now as Rust core uses 'patch_size: usize'

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

    if mode == "streak":
        # Multiscale Strategy: Laplacian Pyramid
        # Decompose image into frequency bands to target streaks of different widths.
        scales = filter_kwargs.get("scales", 3)
        strength = filter_kwargs.get("streak_strength", 1.0)
        
        # Manual Decomposition to safely handle ranges
        from skimage.transform import pyramid_reduce, pyramid_expand
        
        # 1. Build Gaussian Pyramid
        gaussians = [z_norm]
        for i in range(scales - 1):
            # Downscale
            g_next = pyramid_reduce(gaussians[-1], downscale=2)
            gaussians.append(g_next)
            
        # 2. Build Laplacian Pyramid
        pyramid = []
        for i in range(scales - 1):
            g_curr = gaussians[i]
            g_next = gaussians[i+1]
            # Expand next to match current
            g_next_expanded = pyramid_expand(g_next, upscale=2, order=1)
            
            # Match shape
            if g_next_expanded.shape != g_curr.shape:
                from skimage.transform import resize
                g_next_expanded = resize(g_next_expanded, g_curr.shape, order=1, mode='reflect', preserve_range=True)
                
            lap = g_curr - g_next_expanded
            pyramid.append(lap)
            
        # Last band is the Gaussian Low Pass
        pyramid.append(gaussians[-1])
        
        denoised_pyramid = []
        
        for i, band in enumerate(pyramid):
            # 1. Construct Streak PSD for this band
            # We estimate the streak component from the band itself.
            # We use the input base sigma for the random component.
            
            # Ensure band is contiguous float32 for Rust
            band_c = np.ascontiguousarray(band, dtype=np.float32)
            
            # Robust estimate on the band
            sigma_map_band = construct_streak_psd(
                (patch_size_dim, patch_size_dim), 
                sigma, # Base random sigma passed as is
                sinogram=band_c, 
                streak_strength=strength
            )
            
            # --- Step 1: HT ---
            yhat_ht = bm3d_rust.bm3d_hard_thresholding(
                band_c, band_c, sigma_map_band, sigma, threshold_hard,
                patch_size_dim, step_size, cut_off, num_patches
            )
            
            # --- Step 2: Wiener ---
            # Reuse HT output as pilot
            yhat_wie = bm3d_rust.bm3d_wiener_filtering(
                 band_c, yhat_ht, sigma_map_band, sigma,
                 patch_size_dim, step_size, cut_off, num_patches
            )
            
            denoised_pyramid.append(yhat_wie)
            
        # Reconstruct
        # Start with the simplified lower frequency band (last element)
        y_est = denoised_pyramid[-1]
        for detail in reversed(denoised_pyramid[:-1]):
            # Expand current guess to match detail scale
            expanded = pyramid_expand(y_est, upscale=2, order=1)
            
            # Handle shape mismatch due to odd dimensions during downscaling
            if expanded.shape != detail.shape:
                er, ec = expanded.shape
                dr, dc = detail.shape
                from skimage.transform import resize
                expanded = resize(expanded, detail.shape, order=1, mode='reflect', preserve_range=True)
            
            y_est = expanded + detail
        
        # Denormalize
        if d_max > d_min:
            y_est = y_est * (d_max - d_min) + d_min
            
        return y_est

    else:
        # Generic Mode (Scalar Sigma)
        # We must now pass a 1x1 array for sigma to match new Rust API
        # And sigma_random
        
        # 1x1 Dummy PSD (ignored by Rust when use_colored_noise=false)
        sigma_psd_dummy = np.zeros((1, 1), dtype=np.float32)
        sigma_norm = sigma
        
        yhat_ht = bm3d_rust.bm3d_hard_thresholding(
            z_norm, z_norm, sigma_psd_dummy, sigma_norm, threshold_hard,
            patch_size_dim, step_size, cut_off, num_patches
        )
        
        z_gft_ht = yhat_ht 
        yhat_ht_gft = bm3d_rust.bm3d_hard_thresholding(
            z_norm, z_gft_ht, sigma_psd_dummy, sigma_norm, threshold_hard,
            patch_size_dim, step_size, cut_off, num_patches
        )
        
        yhat_wie = bm3d_rust.bm3d_wiener_filtering(
             z_norm, yhat_ht_gft, sigma_psd_dummy, sigma_norm,
             patch_size_dim, step_size, cut_off, num_patches
        )
        
        z_gft_wie = yhat_wie
        yhat_final = bm3d_rust.bm3d_wiener_filtering(
             z_norm, z_gft_wie, sigma_psd_dummy, sigma_norm,
             patch_size_dim, step_size, cut_off, num_patches
        )
    
    # Restore original range
    if d_max > d_min:
        yhat_final = yhat_final * (d_max - d_min) + d_min
    
    return yhat_final
