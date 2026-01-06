
import numpy as np
import matplotlib.pyplot as plt
import sys
from unittest.mock import MagicMock

# Mock cupy/numba if needed (same as bm3d.py)
sys.modules["cupy"] = MagicMock()
sys.modules["numba"] = MagicMock()
sys.modules["numba.cuda"] = MagicMock()

from bm3dornl import bm3d

def create_synthetic_data(shape=(256, 256)):
    """Create a phantom with horizontal structures + vertical streaks."""
    rows, cols = shape
    
    # 1. Ground Truth: Horizontal Sine Waves (should be preserved)
    # Freq in Y direction varies
    y, x = np.mgrid[0:rows, 0:cols]
    gt = np.sin(2 * np.pi * y / 20.0) * 0.5 + 0.5 # 0..1
    
    # 2. Add Vertical Streaks (Static Noise)
    # Varies in X, constant in Y
    np.random.seed(42)
    streak_profile = np.random.normal(0, 0.1, size=cols) # Sigma=0.1 streaks
    streak_profile = np.convolve(streak_profile, np.ones(3)/3, mode='same') # Smooth slightly in X
    streaks = np.tile(streak_profile, (rows, 1))
    
    # 3. Add Random White Noise
    random_noise = np.random.normal(0, 0.02, size=(rows, cols)) # Sigma=0.02 random
    
    noisy = gt + streaks + random_noise
    
    return gt, noisy, streaks, random_noise

def verify_streak_removal():
    print("Running Synthetic Streak Removal Verification...")
    H, W = 256, 256
    gt, noisy, streaks, random_noise = create_synthetic_data((H, W))
    
    # Run Streak Removal
    # We estimate sigma_random approx 0.02
    sigma_input = 0.02
    
    print(f"Input PSNR: {10 * np.log10(1.0 / np.mean((gt - noisy)**2)):.2f} dB")
    
    # Run Streak Removal
    # We estimate sigma_random approx 0.02
    sigma_input = 0.02
    
    # Enable exposure of streak_strength in bm3d_ring_artifact_removal if not already?
    # Actually, I need to modify bm3d.py to accept streak_strength in filter_kwargs or main args.
    # For now, let's just assert that *if* we pass it, it works. 
    # But bm3d.py currently handles it inside:
    # sigma_map = construct_streak_psd(..., streak_strength=1.0)
    # I MUST update bm3d.py to read streak_strength from kwargs.
    
    filter_kwargs = {"streak_strength": 5.0} # Aggressive
    
    denoised = bm3d.bm3d_ring_artifact_removal(
        noisy, 
        mode="streak", 
        sigma=sigma_input,
        block_matching_kwargs={"patch_size": (8, 8), "stride": 4},
        filter_kwargs=filter_kwargs
    )
    
    # Metrics
    mse = np.mean((gt - denoised)**2)
    psnr = 10 * np.log10(1.0 / mse)
    
    print(f"Denoised PSNR: {psnr:.2f} dB")
    
    # Check Streak Removal Efficiency
    # Vertical profile of residual should be low
    residual = noisy - denoised
    # Ideally residual contains Streaks + Random Noise
    # We want denoised ~ gt
    
    # Visual check logic:
    # 1. Did we remove streaks? (Mean vertical profile of denoised should be close to Mean vertical profile of GT)
    prof_gt = np.mean(gt, axis=0) # Should be const 0.5 approx? No, sin(y) averages to 0.5.
    prof_denoised = np.mean(denoised, axis=0)
    prof_noisy = np.mean(noisy, axis=0)
    
    streak_rem_mse = np.mean((prof_gt - prof_denoised)**2)
    streak_in_mse = np.mean((prof_gt - prof_noisy)**2)
    
    print(f"Streak Component MSE (Input):    {streak_in_mse:.6f}")
    print(f"Streak Component MSE (Denoised): {streak_rem_mse:.6f}")
    
    if streak_rem_mse < streak_in_mse * 0.1:
        print("PASS: Significant streak reduction.")
    else:
        print("FAIL: Streaks persist.")
        
    # 2. Did we preserve horizontal structure?
    # By checking PSNR, we already check this somewhat.
    # But let's check high freq in Y.
    
    if psnr > 30.0:
        print("PASS: High PSNR achieved (Structure preserved).")
    else:
        print("FAIL: Low PSNR (Likely over-smoothed).")

if __name__ == "__main__":
    verify_streak_removal()
