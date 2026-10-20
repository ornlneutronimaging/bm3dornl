//! Rust implementation of BM3D (Block-Matching and 3D filtering) algorithm
//! for image denoising.
//!
//! The library supports both f32 and f64 precision through the `Bm3dFloat` trait.

pub mod block_matching;
pub mod filtering;
pub mod float_trait;
pub mod fourier_svd;
pub mod multiscale;
pub mod noise_estimation;
pub mod orchestration;
pub mod pipeline;
pub mod streak;
pub mod transforms;
pub mod utils;

pub use block_matching::PatchMatch;
pub use float_trait::Bm3dFloat;
pub use multiscale::{
    MultiscaleConfig, multiscale_bm3d_streak_removal, multiscale_bm3d_streak_removal_with_plans,
};
pub use noise_estimation::estimate_noise_sigma;
pub use orchestration::{Bm3dConfig, RingRemovalMode, bm3d_ring_artifact_removal};
pub use pipeline::{
    Bm3dKernelConfig, Bm3dMode, Bm3dPlans, run_bm3d_kernel, run_bm3d_step, run_bm3d_step_stack,
    set_use_hadamard_fast_path, use_hadamard_fast_path,
};
pub use streak::estimate_streak_profile_impl;
pub use transforms::{fft2d, ifft2d, wht2d_8x8_forward, wht2d_8x8_inverse};
