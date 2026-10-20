//! Images smaller than `patch_size`.
//!
//! `bm3d_ring_artifact_removal` refuses such an image with "Image size
//! (rows, cols) is smaller than patch_size p".
//! `run_bm3d_step` and `run_bm3d_step_stack` must return the same error.
//! Multiscale processing must return it too. An image exactly `patch_size` in
//! a dimension is valid.

use std::panic::{AssertUnwindSafe, catch_unwind};

use bm3d_core::{
    Bm3dConfig, Bm3dFloat, Bm3dKernelConfig, Bm3dMode, Bm3dPlans, MultiscaleConfig,
    RingRemovalMode, bm3d_ring_artifact_removal, multiscale_bm3d_streak_removal, run_bm3d_step,
    run_bm3d_step_stack,
};
use ndarray::{Array2, Array3};

const PATCH_SIZE: usize = 8;
const TOO_SMALL: [(usize, usize); 3] = [(6, 64), (64, 6), (7, 7)];
const AT_PATCH_SIZE: [(usize, usize); 3] = [(8, 64), (64, 8), (8, 8)];
const MULTISCALE_TOO_SMALL: [(usize, usize); 5] = [(7, 200), (200, 7), (6, 64), (64, 6), (7, 7)];
const MULTISCALE_AT_PATCH_SIZE: [(usize, usize); 2] = [(8, 200), (200, 8)];
const KERNEL_MODES: [Bm3dMode; 2] = [Bm3dMode::HardThreshold, Bm3dMode::Wiener];
const RING_MODES: [RingRemovalMode; 3] = [
    RingRemovalMode::Generic,
    RingRemovalMode::Streak,
    RingRemovalMode::FourierSvd,
];

fn noisy_image<F: Bm3dFloat>(rows: usize, cols: usize, seed: u64) -> Array2<F> {
    let mut state = seed;
    Array2::from_shape_fn((rows, cols), |(_, c)| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = (state >> 40) as f64 / (1u64 << 24) as f64 - 0.5;
        let truth = if c >= 41 { 0.75 } else { 0.4 };
        F::from_f64_c(truth + 0.1 * noise)
    })
}

fn noisy_stack<F: Bm3dFloat>(slices: usize, rows: usize, cols: usize) -> Array3<F> {
    let mut stack = Array3::zeros((slices, rows, cols));
    for (i, mut slice) in stack.outer_iter_mut().enumerate() {
        slice.assign(&noisy_image::<F>(rows, cols, 11 + i as u64));
    }
    stack
}

fn too_small_message(rows: usize, cols: usize, patch_size: usize) -> String {
    format!("Image size ({rows}, {cols}) is smaller than patch_size {patch_size}")
}

fn kernel_config<F: Bm3dFloat>(patch_size: usize) -> Bm3dKernelConfig<F> {
    Bm3dKernelConfig {
        sigma_random: F::from_f64_c(0.05),
        threshold: F::from_f64_c(2.7),
        patch_size,
        step_size: 4,
        search_window: 24,
        max_matches: 16,
        search_step: 4,
        use_hadamard_fast_path: None,
    }
}

fn without_panic<T>(label: &str, f: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(f))
        .unwrap_or_else(|_| panic!("{label} panicked instead of returning a Result"))
}

fn step_2d<F: Bm3dFloat>(
    rows: usize,
    cols: usize,
    mode: Bm3dMode,
    patch_size: usize,
) -> Result<Array2<F>, String> {
    let image = noisy_image::<F>(rows, cols, 5);
    let sigma_psd = Array2::<F>::zeros((1, 1));
    let sigma_map = Array2::<F>::zeros((1, 1));
    let plans = Bm3dPlans::new(patch_size, 16);
    run_bm3d_step(
        image.view(),
        image.view(),
        mode,
        sigma_psd.view(),
        sigma_map.view(),
        &kernel_config(patch_size),
        &plans,
    )
}

fn step_stack<F: Bm3dFloat>(
    rows: usize,
    cols: usize,
    mode: Bm3dMode,
    patch_size: usize,
) -> Result<Array3<F>, String> {
    let stack = noisy_stack::<F>(2, rows, cols);
    let sigma_psd = Array2::<F>::zeros((1, 1));
    let sigma_map = Array3::<F>::zeros((1, 1, 1));
    let plans = Bm3dPlans::new(patch_size, 16);
    run_bm3d_step_stack(
        stack.view(),
        stack.view(),
        mode,
        sigma_psd.view(),
        sigma_map.view(),
        &kernel_config(patch_size),
        &plans,
        None,
    )
}

fn assert_too_small<T>(label: &str, result: Result<T, String>, rows: usize, cols: usize, p: usize) {
    let expected = too_small_message(rows, cols, p);
    match result {
        Ok(_) => panic!("{label}: accepted a {rows} x {cols} image at patch_size {p}"),
        Err(err) => assert!(
            err.contains(&expected),
            "{label}: expected an error containing {expected:?}, got {err:?}"
        ),
    }
}

fn check_step_rejects_small_images<F: Bm3dFloat>(precision: &str) {
    for (rows, cols) in TOO_SMALL {
        for mode in KERNEL_MODES {
            let label = format!("run_bm3d_step {precision} {mode:?} {rows}x{cols}");
            let result = without_panic(&label, || step_2d::<F>(rows, cols, mode, PATCH_SIZE));
            assert_too_small(&label, result, rows, cols, PATCH_SIZE);
        }
    }
}

#[test]
fn run_bm3d_step_rejects_images_smaller_than_patch_size() {
    check_step_rejects_small_images::<f32>("f32");
    check_step_rejects_small_images::<f64>("f64");
}

fn check_stack_rejects_small_images<F: Bm3dFloat>(precision: &str) {
    for (rows, cols) in TOO_SMALL {
        for mode in KERNEL_MODES {
            let label = format!("run_bm3d_step_stack {precision} {mode:?} 2x{rows}x{cols}");
            let result = without_panic(&label, || step_stack::<F>(rows, cols, mode, PATCH_SIZE));
            assert_too_small(&label, result, rows, cols, PATCH_SIZE);
        }
    }
}

#[test]
fn run_bm3d_step_stack_rejects_images_smaller_than_patch_size() {
    check_stack_rejects_small_images::<f32>("f32");
    check_stack_rejects_small_images::<f64>("f64");
}

#[test]
fn kernel_check_uses_the_configured_patch_size() {
    for mode in KERNEL_MODES {
        let label = format!("run_bm3d_step f32 {mode:?} 8x64 at patch_size 9");
        let result = without_panic(&label, || step_2d::<f32>(8, 64, mode, 9));
        assert_too_small(&label, result, 8, 64, 9);
        let label = format!("run_bm3d_step_stack f64 {mode:?} 2x8x64 at patch_size 9");
        let result = without_panic(&label, || step_stack::<f64>(8, 64, mode, 9));
        assert_too_small(&label, result, 8, 64, 9);
    }
}

fn check_kernels_accept_images_at_patch_size<F: Bm3dFloat>(precision: &str) {
    for (rows, cols) in AT_PATCH_SIZE {
        for mode in KERNEL_MODES {
            let out = step_2d::<F>(rows, cols, mode, PATCH_SIZE).unwrap_or_else(|e| {
                panic!("run_bm3d_step {precision} {mode:?} {rows}x{cols} failed: {e}")
            });
            assert_eq!(out.dim(), (rows, cols));
            assert!(out.iter().all(|v| v.is_finite()));
            let out = step_stack::<F>(rows, cols, mode, PATCH_SIZE).unwrap_or_else(|e| {
                panic!("run_bm3d_step_stack {precision} {mode:?} 2x{rows}x{cols} failed: {e}")
            });
            assert_eq!(out.dim(), (2, rows, cols));
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }
}

#[test]
fn kernels_accept_images_exactly_patch_size() {
    check_kernels_accept_images_at_patch_size::<f32>("f32");
    check_kernels_accept_images_at_patch_size::<f64>("f64");
}

fn check_ring_removal_rejects_small_images<F: Bm3dFloat>(precision: &str) {
    let config = Bm3dConfig::<F>::default();
    for (rows, cols) in TOO_SMALL {
        let image = noisy_image::<F>(rows, cols, 3);
        for mode in RING_MODES {
            let label = format!("bm3d_ring_artifact_removal {precision} {mode:?} {rows}x{cols}");
            let result = without_panic(&label, || {
                bm3d_ring_artifact_removal(image.view(), mode, &config)
            });
            assert_too_small(&label, result, rows, cols, PATCH_SIZE);
        }
    }
}

#[test]
fn ring_removal_rejects_images_smaller_than_patch_size() {
    check_ring_removal_rejects_small_images::<f32>("f32");
    check_ring_removal_rejects_small_images::<f64>("f64");
}

fn check_multiscale_rejects_small_images<F: Bm3dFloat>(precision: &str) {
    for log_domain_input in [false, true] {
        let config = MultiscaleConfig::<F> {
            log_domain_input,
            ..MultiscaleConfig::default()
        };
        for (rows, cols) in MULTISCALE_TOO_SMALL {
            let image = noisy_image::<F>(rows, cols, 9);
            let label =
                format!("multiscale {precision} log_domain_input={log_domain_input} {rows}x{cols}");
            let result = without_panic(&label, || {
                multiscale_bm3d_streak_removal(image.view(), &config)
            });
            assert_too_small(&label, result, rows, cols, PATCH_SIZE);
        }
    }
}

#[test]
fn multiscale_rejects_images_smaller_than_patch_size() {
    check_multiscale_rejects_small_images::<f32>("f32");
    check_multiscale_rejects_small_images::<f64>("f64");
}

fn check_multiscale_accepts_images_at_patch_size<F: Bm3dFloat>(precision: &str) {
    let config = MultiscaleConfig::<F>::default();
    for (rows, cols) in MULTISCALE_AT_PATCH_SIZE {
        let image = noisy_image::<F>(rows, cols, 9);
        let out = multiscale_bm3d_streak_removal(image.view(), &config)
            .unwrap_or_else(|e| panic!("multiscale {precision} {rows}x{cols} failed: {e}"));
        assert_eq!(out.dim(), (rows, cols));
        assert!(out.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn multiscale_accepts_images_exactly_patch_size() {
    check_multiscale_accepts_images_at_patch_size::<f32>("f32");
    check_multiscale_accepts_images_at_patch_size::<f64>("f64");
}
