import numpy as np
from scipy.ndimage import gaussian_filter
import skimage.metrics
from bm3dornl.bm3d import bm3d_ring_artifact_removal


def generate_synthetic_sinogram(size=(256, 256), snr_db=10):
    """Generates a synthetic sinogram with vertical streaks (ring artifacts)."""
    # 1. Create a clean sinogram (smooth features)
    x = np.linspace(-1, 1, size[1])
    y = np.linspace(-1, 1, size[0])
    X, Y = np.meshgrid(x, y)

    # Simple phantom: A few Gaussian blobs
    clean = np.zeros(size)
    clean += 0.5 * np.exp(-((X) ** 2 + (Y) ** 2) / 0.1)
    clean += 0.3 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.05)

    # 2. Add Streaks (Vertical lines)
    np.random.seed(42)
    streak_profile = (np.random.rand(size[1]) - 0.5) * 0.2
    streak_profile = gaussian_filter(streak_profile, 1.0)
    streaks = np.tile(streak_profile, (size[0], 1))

    # 3. Add Noise
    signal_power = np.mean(clean**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_sigma = np.sqrt(noise_power)

    noise = np.random.normal(0, noise_sigma, size)

    noisy_streaky = clean + streaks + noise

    return clean, noisy_streaky, noise_sigma


def test_streak_mode_improvement():
    """Verifies that Streak Mode provides significant improvement over Generic BM3D on streaky data."""
    clean, noisy, sigma = generate_synthetic_sinogram(size=(512, 512), snr_db=5)

    # Generic
    denoised_generic = bm3d_ring_artifact_removal(noisy, mode="generic", sigma=sigma)
    psnr_generic = skimage.metrics.peak_signal_noise_ratio(clean, denoised_generic)

    import time

    start = time.time()
    # Streak
    denoised_streak = bm3d_ring_artifact_removal(noisy, mode="streak", sigma=sigma)
    end = time.time()
    psnr_streak = skimage.metrics.peak_signal_noise_ratio(clean, denoised_streak)

    print(f"Time: {end - start:.4f} seconds")
    print(f"PSNR: {psnr_streak:.2f} dB")

    diff = psnr_streak - psnr_generic
    print(
        f"Generic: {psnr_generic:.2f} dB, Streak: {psnr_streak:.2f} dB, Diff: {diff:.2f} dB"
    )

    # Expect at least 3 dB improvement (based on successful fix)
    assert diff > 3.0, (
        f"Streak mode improvement ({diff:.2f} dB) is insufficient (< 3.0 dB)"
    )


if __name__ == "__main__":
    test_streak_mode_improvement()
