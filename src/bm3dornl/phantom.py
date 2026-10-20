#!/usr/bin/env python3

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import radon
from typing import Tuple


def shepp_logan_phantom(size: int = 256, contrast_factor: float = 2.0) -> np.ndarray:
    """
    Generate a Shepp-Logan phantom with intensity values normalized between 0 and 1.

    Parameters
    ----------
    size : int, optional
        The width and height of the square image, by default 256.
    """
    ellipses = [
        [0.69, 0.92, 0, 0, 0, 2],
        [0.6624, 0.874, 0, -0.0184, 0, -0.98],
        [0.21, 0.25, 0.22, 0, -18, -0.2],
        [0.16, 0.41, -0.22, 0, 18, -0.2],
        [0.21, 0.25, 0, 0.35, 0, 0.1],
        [0.046, 0.046, 0, 0.1, 0, 0.2],
        [0.046, 0.046, 0, -0.1, 0, 0.2],
        [0.046, 0.023, -0.08, -0.605, 0, 0.2],
        [0.023, 0.023, 0, -0.606, 0, 0.2],
        [0.023, 0.046, 0.06, -0.605, 0, 0.2],
    ]

    phantom = np.zeros((size, size))

    for ellipse in ellipses:
        a, b, x0, y0, phi, intensity = ellipse
        intensity *= contrast_factor
        y, x = np.ogrid[-1 : 1 : size * 1j, -1 : 1 : size * 1j]
        phi = np.deg2rad(phi)
        x_rot = x * np.cos(phi) + y * np.sin(phi)
        y_rot = -x * np.sin(phi) + y * np.cos(phi)
        mask = ((x_rot - x0) ** 2 / a**2) + ((y_rot - y0) ** 2 / b**2) <= 1
        phantom += mask * intensity

    min_val = phantom.min()
    max_val = phantom.max()
    phantom = (phantom - min_val) / (max_val - min_val)

    return phantom


def generate_sinogram(
    input_img: np.ndarray,
    scan_step: float,
) -> Tuple[np.ndarray, np.ndarray]:
    thetas_deg = np.arange(-180, 180, scan_step)

    sinogram = radon(
        input_img,
        theta=thetas_deg,
        circle=False,
    ).T

    sinogram = (sinogram - sinogram.min()) / (sinogram.max() - sinogram.min()) + 1e-8

    return sinogram, thetas_deg


def simulate_detector_gain_error(
    sinogram: np.ndarray,
    detector_gain_range: Tuple[float, float],
    detector_gain_error: float,
) -> Tuple[np.ndarray, np.ndarray]:
    detector_gain = np.random.uniform(
        detector_gain_range[0],
        detector_gain_range[1],
        sinogram.shape[1],
    )
    detector_gain = np.ones(sinogram.shape) * detector_gain

    if detector_gain_error != 0.0:
        detector_gain = np.random.normal(
            detector_gain,
            detector_gain * detector_gain_error,
        )

    sinogram = sinogram * detector_gain

    sinogram = (sinogram - sinogram.min()) / (sinogram.max() - sinogram.min()) + 1e-8

    sinogram = sinogram.astype(np.float32)
    detector_gain = detector_gain.astype(np.float32)

    return sinogram, detector_gain


def apply_convolution(noise: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    noise_fft = np.fft.fftn(noise)
    kernel_fft = np.fft.fftn(kernel, s=noise.shape)
    convolved_noise = np.real(np.fft.ifftn(noise_fft * kernel_fft))
    return convolved_noise


def get_synthetic_noise(
    image_size: Tuple[int, int],
    streak_kernel_width: int,
    streak_kernel_length: int,
    white_noise_intensity: float,
    streak_noise_intensity: float,
) -> np.ndarray:
    """
    Generate synthetic noise composed of white noise and streak noise.

    Parameters
    ----------
    image_size : Tuple[int, int]
        The size of the noise image (height, width).
    """
    mu = np.random.normal(0, 1, image_size)

    g0 = np.zeros(image_size)
    g0[image_size[0] // 2, image_size[1] // 2] = 1
    white_noise = apply_convolution(mu, g0) * white_noise_intensity

    g1 = np.zeros(image_size)
    g1[image_size[0] // 2, image_size[1] // 2] = 1
    g1 = gaussian_filter(g1, sigma=(streak_kernel_length, streak_kernel_width))
    streak_noise = apply_convolution(mu, g1) * streak_noise_intensity

    return white_noise + streak_noise
