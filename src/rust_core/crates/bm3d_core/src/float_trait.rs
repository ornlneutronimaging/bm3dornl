//! Float trait abstraction for f32/f64 support.
//!
//! This module provides a unified trait for floating-point operations,
//! enabling the BM3D library to work with both f32 and f64 precision.

use num_traits::{Float, FromPrimitive, NumAssign};
use rustfft::FftNum;
use std::fmt::Debug;
use std::iter::Sum;

/// Trait alias for floating point types supported by BM3D.
///
/// This trait combines all the bounds needed for BM3D operations:
/// - Basic float operations (Float, NumAssign)
/// - FFT compatibility (FftNum from rustfft)
/// - Conversion from primitive types (FromPrimitive)
/// - Iteration support (Sum)
/// - Debug printing
pub trait Bm3dFloat:
    Float + FftNum + FromPrimitive + NumAssign + Sum + Debug + Send + Sync + 'static
{
    /// The constant PI for this float type.
    const PI: Self;

    /// The constant for Gaussian truncation factor (4.0).
    const GAUSSIAN_TRUNCATE: Self;

    /// Create a value from an f64 constant.
    fn from_f64_c(val: f64) -> Self;

    /// Create a value from a usize constant.
    fn usize_as(val: usize) -> Self;

    /// Create a value from an isize constant.
    fn isize_as(val: isize) -> Self;
}

impl Bm3dFloat for f32 {
    const PI: Self = std::f32::consts::PI;
    const GAUSSIAN_TRUNCATE: Self = 4.0;

    #[inline]
    fn from_f64_c(val: f64) -> Self {
        val as f32
    }

    #[inline]
    fn usize_as(val: usize) -> Self {
        val as f32
    }

    #[inline]
    fn isize_as(val: isize) -> Self {
        val as f32
    }
}

impl Bm3dFloat for f64 {
    const PI: Self = std::f64::consts::PI;
    const GAUSSIAN_TRUNCATE: Self = 4.0;

    #[inline]
    fn from_f64_c(val: f64) -> Self {
        val
    }

    #[inline]
    fn usize_as(val: usize) -> Self {
        val as f64
    }

    #[inline]
    fn isize_as(val: isize) -> Self {
        val as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_trait_impl() {
        let val: f32 = Bm3dFloat::from_f64_c(std::f64::consts::PI);
        assert!((val - std::f32::consts::PI).abs() < 1e-5);

        let usize_val: f32 = Bm3dFloat::usize_as(42);
        assert_eq!(usize_val, 42.0f32);

        let isize_val: f32 = Bm3dFloat::isize_as(-5);
        assert_eq!(isize_val, -5.0f32);
    }

    #[test]
    fn test_f64_trait_impl() {
        let val: f64 = Bm3dFloat::from_f64_c(std::f64::consts::PI);
        assert!((val - std::f64::consts::PI).abs() < 1e-14);

        let usize_val: f64 = Bm3dFloat::usize_as(42);
        assert_eq!(usize_val, 42.0f64);

        let isize_val: f64 = Bm3dFloat::isize_as(-5);
        assert_eq!(isize_val, -5.0f64);
    }

    #[test]
    fn test_pi_constants() {
        assert!((f32::PI - std::f32::consts::PI).abs() < 1e-10);
        assert!((f64::PI - std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn test_gaussian_truncate() {
        assert_eq!(f32::GAUSSIAN_TRUNCATE, 4.0f32);
        assert_eq!(f64::GAUSSIAN_TRUNCATE, 4.0f64);
    }
}
