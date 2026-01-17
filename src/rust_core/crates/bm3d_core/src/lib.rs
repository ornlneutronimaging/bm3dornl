//! BM3D Core Algorithm Library
//!
//! Pure Rust implementation of BM3D (Block-Matching and 3D filtering) algorithm
//! for image denoising. This crate contains all algorithm logic without Python bindings.
//!
//! ## f32/f64 Support
//!
//! The library supports both f32 and f64 precision through the `Bm3dFloat` trait.
//! All public functions are generic over this trait, allowing users to choose
//! the precision that best fits their needs.

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

// Re-export commonly used types at the crate root
pub use block_matching::PatchMatch;
pub use float_trait::Bm3dFloat;
pub use multiscale::{multiscale_bm3d_streak_removal, MultiscaleConfig};
pub use noise_estimation::estimate_noise_sigma;
pub use orchestration::{bm3d_ring_artifact_removal, Bm3dConfig, RingRemovalMode};
pub use pipeline::{run_bm3d_kernel, run_bm3d_step, run_bm3d_step_stack, Bm3dMode, Bm3dPlans};
pub use streak::estimate_streak_profile_impl;
pub use transforms::{fft2d, ifft2d, wht2d_8x8_forward, wht2d_8x8_inverse};
