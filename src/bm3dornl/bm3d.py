#!/usr/bin/env python3
"""Denoising functions using Rust backend."""

import logging
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
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

def fft_streak_removal(sinogram, sigma_y=1.0, sigma_protect_x=20.0):
    """
    Remove vertical streaks using a FFT-based Notch Filter.
    
    Streaks correspond to the DC component in the vertical frequency axis (v_y = 0).
    We dampen this axis using a Gaussian notch, while protecting the low-frequency 
    horizontal components (centered at v_x=0) to preserve the object's mean structure.
    """
    H, W = sinogram.shape
    
    # 1. FFT
    f = fft2(sinogram)
    fshift = fftshift(f)
    
    # 2. Construct Filter
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[-cy:H-cy, -cx:W-cx]
    
    # Gaussian Notch along v_y
    notch_y = np.exp(-(y**2) / (2 * sigma_y**2))
    
    # Protection Mask along v_x (Center)
    protect_x = np.exp(-(x**2) / (2 * sigma_protect_x**2))
    
    # Combined Mask: 1 - Notch + Notch*Protection
    # If Protection=1 (Center), Mask=1 (No filtering).
    # If Protection=0 (High Freq X), Mask=1-Notch (Filter out v_y=0).
    mask = 1.0 - notch_y * (1.0 - protect_x)
    
    # 3. Apply
    fshift_filtered = fshift * mask
    
    # 4. IFFT
    img_filtered = np.real(ifft2(ifftshift(fshift_filtered)))
    
    return img_filtered.astype(np.float32)

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
        "streak": FFT-based Streak Removal + Standard BM3D.
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

    # --- Mode Support ---
    if mode == "streak":
        # Spectral Filtering for Streaks (FFT-based)
        sigma_notch = filter_kwargs.get("fft_notch_sigma", 3.0)
        sigma_protect = filter_kwargs.get("fft_protect_width", 25.0)
        
        # Apply FFT Denoising IN-PLACE on z_norm (conceptually)
        z_norm = fft_streak_removal(z_norm, sigma_notch, sigma_protect)
        
        # After removing streaks, the image typically still has random noise.
        # We proceed to run Generic BM3D on this "streak-free" base.

    # --- Generic BM3D Execution ---
    # (Used for both 'generic' mode and the post-FFT 'streak' mode)
    
    # Generic Mode (Scalar Sigma)
    sigma_psd_dummy = np.zeros((1, 1), dtype=np.float32)
    sigma_norm = sigma
    
    # Step 1: Hard Thresholding
    yhat_ht = bm3d_rust.bm3d_hard_thresholding(
        z_norm, z_norm, sigma_psd_dummy, sigma_norm, threshold_hard,
        patch_size_dim, step_size, cut_off, num_patches
    )
    
    z_gft_ht = yhat_ht 
    yhat_ht_gft = bm3d_rust.bm3d_hard_thresholding(
        z_norm, z_gft_ht, sigma_psd_dummy, sigma_norm, threshold_hard,
        patch_size_dim, step_size, cut_off, num_patches
    )
    
    # Step 2: Wiener Filtering
    yhat_wie = bm3d_rust.bm3d_wiener_filtering(
         z_norm, yhat_ht_gft, sigma_psd_dummy, sigma_norm,
         patch_size_dim, step_size, cut_off, num_patches
    )
    
    z_gft_wie = yhat_wie
    yhat_final = bm3d_rust.bm3d_wiener_filtering(
         z_norm, z_gft_wie, sigma_psd_dummy, sigma_norm,
         patch_size_dim, step_size, cut_off, num_patches
    )
    
    # Denormalize and Return
    if d_max > d_min:
        yhat_final = yhat_final * (d_max - d_min) + d_min
    return yhat_final
