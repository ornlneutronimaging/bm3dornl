//! `step_size` must not exceed `patch_size`.
//!
//! Reference patches sit `step_size` apart. With a step larger than the patch
//! they leave gaps. Such a step is refused with "step_size (s) must not
//! exceed patch_size (p)": by `Bm3dConfig::validate`, so by every entry point
//! that validates a config, and by the kernel entry points `run_bm3d_step`
//! and `run_bm3d_step_stack`. A step equal to `patch_size` is valid and
//! filters every pixel. The kernel entry points also return an error for a
//! `step_size` or `search_step` of 0.

use std::panic::{AssertUnwindSafe, catch_unwind};

use bm3d_core::{
    Bm3dConfig, Bm3dFloat, Bm3dKernelConfig, Bm3dMode, Bm3dPlans, MultiscaleConfig,
    RingRemovalMode, bm3d_ring_artifact_removal, multiscale_bm3d_streak_removal, run_bm3d_step,
    run_bm3d_step_stack,
};
use ndarray::{Array2, Array3, Axis};

/// (patch_size, step_size) pairs with the step larger than the patch.
const TOO_LARGE: [(usize, usize); 4] = [(8, 9), (8, 12), (8, 16), (7, 8)];
/// (patch_size, step_size) pairs that stay valid.
const VALID: [(usize, usize); 4] = [(8, 1), (8, 4), (8, 8), (7, 7)];
const KERNEL_MODES: [Bm3dMode; 2] = [Bm3dMode::HardThreshold, Bm3dMode::Wiener];
const RING_MODES: [RingRemovalMode; 3] = [
    RingRemovalMode::Generic,
    RingRemovalMode::Streak,
    RingRemovalMode::FourierSvd,
];
const ROWS: usize = 64;
const COLS: usize = 80;

/// Positive step edges plus uniform noise from a fixed-seed linear
/// congruential generator.
fn noisy_image<F: Bm3dFloat>(rows: usize, cols: usize, seed: u64) -> Array2<F> {
    let mut state = seed;
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = (state >> 40) as f64 / (1u64 << 24) as f64 - 0.5;
        let truth = if c >= 41 { 0.75 } else { 0.25 } + if r >= 35 { 0.15 } else { 0.0 };
        F::from_f64_c(truth + 0.2 * noise)
    })
}

fn noisy_stack<F: Bm3dFloat>(slices: usize) -> Array3<F> {
    let mut stack = Array3::zeros((slices, ROWS, COLS));
    for (i, mut slice) in stack.outer_iter_mut().enumerate() {
        slice.assign(&noisy_image::<F>(ROWS, COLS, 21 + i as u64));
    }
    stack
}

fn too_large_message(patch_size: usize, step_size: usize) -> String {
    format!("step_size ({step_size}) must not exceed patch_size ({patch_size})")
}

fn kernel_config<F: Bm3dFloat>(
    patch_size: usize,
    step_size: usize,
    search_step: usize,
) -> Bm3dKernelConfig<F> {
    Bm3dKernelConfig {
        sigma_random: F::from_f64_c(0.05),
        threshold: F::from_f64_c(2.7),
        patch_size,
        step_size,
        search_window: 24,
        max_matches: 16,
        search_step,
        use_hadamard_fast_path: None,
    }
}

fn ring_config<F: Bm3dFloat>(patch_size: usize, step_size: usize) -> Bm3dConfig<F> {
    Bm3dConfig {
        sigma_random: F::from_f64_c(0.07),
        patch_size,
        step_size,
        ..Bm3dConfig::default()
    }
}

/// Run `f` and fail with a clear message if it panics.
fn without_panic<T>(label: &str, f: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(f))
        .unwrap_or_else(|_| panic!("{label} panicked instead of returning a Result"))
}

fn step_2d<F: Bm3dFloat>(
    image: &Array2<F>,
    mode: Bm3dMode,
    config: &Bm3dKernelConfig<F>,
) -> Result<Array2<F>, String> {
    let sigma_psd = Array2::<F>::zeros((1, 1));
    let sigma_map = Array2::<F>::zeros((1, 1));
    let plans = Bm3dPlans::new(config.patch_size, config.max_matches);
    run_bm3d_step(
        image.view(),
        image.view(),
        mode,
        sigma_psd.view(),
        sigma_map.view(),
        config,
        &plans,
    )
}

fn step_stack<F: Bm3dFloat>(
    stack: &Array3<F>,
    mode: Bm3dMode,
    config: &Bm3dKernelConfig<F>,
) -> Result<Array3<F>, String> {
    let sigma_psd = Array2::<F>::zeros((1, 1));
    let sigma_map = Array3::<F>::zeros((1, 1, 1));
    let plans = Bm3dPlans::new(config.patch_size, config.max_matches);
    run_bm3d_step_stack(
        stack.view(),
        stack.view(),
        mode,
        sigma_psd.view(),
        sigma_map.view(),
        config,
        &plans,
        None,
    )
}

fn assert_err_contains<T>(label: &str, result: Result<T, String>, expected: &str) {
    match result {
        Ok(_) => panic!("{label}: accepted, expected an error containing {expected:?}"),
        Err(err) => assert!(
            err.contains(expected),
            "{label}: expected an error containing {expected:?}, got {err:?}"
        ),
    }
}

/// Number of output pixels exactly equal to their input pixel. The kernel
/// returns the input value where no reference patch covers a pixel.
fn unchanged_pixels<F: Bm3dFloat>(output: &Array2<F>, input: &Array2<F>) -> usize {
    output
        .iter()
        .zip(input.iter())
        .filter(|(o, i)| o == i)
        .count()
}

fn check_validate<F: Bm3dFloat>() {
    for (patch_size, step_size) in TOO_LARGE {
        let label = format!("Bm3dConfig::validate patch {patch_size} step {step_size}");
        let config = ring_config::<F>(patch_size, step_size);
        assert_err_contains(
            &label,
            config.validate(),
            &too_large_message(patch_size, step_size),
        );
        let multiscale = MultiscaleConfig::<F> {
            bm3d_config: config,
            ..MultiscaleConfig::default()
        };
        assert_err_contains(
            &format!("MultiscaleConfig::validate patch {patch_size} step {step_size}"),
            multiscale.validate(),
            &too_large_message(patch_size, step_size),
        );
    }
    for (patch_size, step_size) in VALID {
        let config = ring_config::<F>(patch_size, step_size);
        assert!(
            config.validate().is_ok(),
            "patch {patch_size} step {step_size} should validate: {:?}",
            config.validate()
        );
        let multiscale = MultiscaleConfig::<F> {
            bm3d_config: config,
            ..MultiscaleConfig::default()
        };
        assert!(
            multiscale.validate().is_ok(),
            "multiscale patch {patch_size} step {step_size} should validate"
        );
    }
}

#[test]
fn validate_refuses_step_size_above_patch_size() {
    check_validate::<f32>();
    check_validate::<f64>();
}

fn check_public_entry_points<F: Bm3dFloat>(precision: &str) {
    let image = noisy_image::<F>(ROWS, 160, 3);
    for (patch_size, step_size) in TOO_LARGE {
        let expected = too_large_message(patch_size, step_size);
        let config = ring_config::<F>(patch_size, step_size);
        for mode in RING_MODES {
            let label = format!(
                "bm3d_ring_artifact_removal {precision} {mode:?} patch {patch_size} step {step_size}"
            );
            assert_err_contains(
                &label,
                bm3d_ring_artifact_removal(image.view(), mode, &config),
                &expected,
            );
        }
        let multiscale = MultiscaleConfig::<F> {
            bm3d_config: config,
            ..MultiscaleConfig::default()
        };
        assert_err_contains(
            &format!("multiscale {precision} patch {patch_size} step {step_size}"),
            multiscale_bm3d_streak_removal(image.view(), &multiscale),
            &expected,
        );
    }
}

#[test]
fn public_entry_points_refuse_step_size_above_patch_size() {
    check_public_entry_points::<f32>("f32");
    check_public_entry_points::<f64>("f64");
}

fn check_kernels_refuse_large_steps<F: Bm3dFloat>(precision: &str) {
    let image = noisy_image::<F>(ROWS, COLS, 5);
    let stack = noisy_stack::<F>(2);
    for (patch_size, step_size) in TOO_LARGE {
        let expected = too_large_message(patch_size, step_size);
        let config = kernel_config::<F>(patch_size, step_size, step_size);
        for mode in KERNEL_MODES {
            let label =
                format!("run_bm3d_step {precision} {mode:?} patch {patch_size} step {step_size}");
            let result = without_panic(&label, || step_2d(&image, mode, &config));
            assert_err_contains(&label, result, &expected);
            let label = format!(
                "run_bm3d_step_stack {precision} {mode:?} patch {patch_size} step {step_size}"
            );
            let result = without_panic(&label, || step_stack(&stack, mode, &config));
            assert_err_contains(&label, result, &expected);
        }
    }
}

#[test]
fn kernels_refuse_step_size_above_patch_size() {
    check_kernels_refuse_large_steps::<f32>("f32");
    check_kernels_refuse_large_steps::<f64>("f64");
}

fn check_step_equal_to_patch_filters_every_pixel<F: Bm3dFloat>(precision: &str) {
    let image = noisy_image::<F>(ROWS, COLS, 5);
    let stack = noisy_stack::<F>(2);
    for patch_size in [8, 7] {
        let config = kernel_config::<F>(patch_size, patch_size, patch_size);
        for mode in KERNEL_MODES {
            let label = format!("{precision} {mode:?} patch {patch_size} step {patch_size}");
            let out = step_2d(&image, mode, &config)
                .unwrap_or_else(|e| panic!("run_bm3d_step {label} failed: {e}"));
            assert_eq!(
                unchanged_pixels(&out, &image),
                0,
                "run_bm3d_step {label}: some pixels came back unfiltered"
            );
            let out = step_stack(&stack, mode, &config)
                .unwrap_or_else(|e| panic!("run_bm3d_step_stack {label} failed: {e}"));
            for (i, slice) in out.axis_iter(Axis(0)).enumerate() {
                let input = stack.index_axis(Axis(0), i).to_owned();
                assert_eq!(
                    unchanged_pixels(&slice.to_owned(), &input),
                    0,
                    "run_bm3d_step_stack {label} slice {i}: some pixels came back unfiltered"
                );
            }
        }
    }
}

#[test]
fn step_size_equal_to_patch_size_filters_every_pixel() {
    check_step_equal_to_patch_filters_every_pixel::<f32>("f32");
    check_step_equal_to_patch_filters_every_pixel::<f64>("f64");
}

fn check_kernels_refuse_zero_strides<F: Bm3dFloat>(precision: &str) {
    let image = noisy_image::<F>(ROWS, COLS, 5);
    let stack = noisy_stack::<F>(2);
    // (step_size, search_step, name the error must contain). With both 0 the
    // error must name step_size.
    for (step_size, search_step, name) in [
        (0, 4, "step_size"),
        (4, 0, "search_step"),
        (0, 0, "step_size"),
    ] {
        let config = kernel_config::<F>(8, step_size, search_step);
        for mode in KERNEL_MODES {
            let label = format!(
                "run_bm3d_step {precision} {mode:?} step_size {step_size} search_step {search_step}"
            );
            let result = without_panic(&label, || step_2d(&image, mode, &config));
            assert_err_contains(&label, result, name);
            let label = format!(
                "run_bm3d_step_stack {precision} {mode:?} step_size {step_size} search_step {search_step}"
            );
            let result = without_panic(&label, || step_stack(&stack, mode, &config));
            assert_err_contains(&label, result, name);
        }
    }
}

#[test]
fn kernels_refuse_zero_step_size_and_search_step() {
    check_kernels_refuse_zero_strides::<f32>("f32");
    check_kernels_refuse_zero_strides::<f64>("f64");
}
