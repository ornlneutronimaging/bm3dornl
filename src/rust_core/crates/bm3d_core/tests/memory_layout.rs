//! Arrays in any memory layout give the result of standard layout.
//!
//! The same holds for the public module functions `transforms::fft2d_into`,
//! `transforms::ifft2d_into`,
//! `transforms::ifft2d_inplace_to_real_with_plan_scratch` and
//! `block_matching::find_similar_patches_in_place_sum`.
//!
//! A non-standard input may take the indexed path instead of a slice fast
//! path. The two paths add in a different order, so their results agree to
//! rounding, not bit for bit. The check is max |got - want| <= tolerance *
//! max |want|, with tolerance 1e-3 (f32) and 1e-12 (f64). In block matching
//! the positions must be the same, and the distances agree to rounding.

use std::fmt::Debug;

use bm3d_core::block_matching::{compute_integral_sum_image, find_similar_patches_in_place_sum};
use bm3d_core::transforms::{fft2d_into, ifft2d_inplace_to_real_with_plan_scratch, ifft2d_into};
use bm3d_core::{
    Bm3dConfig, Bm3dFloat, Bm3dKernelConfig, Bm3dMode, Bm3dPlans, MultiscaleConfig, PatchMatch,
    RingRemovalMode, bm3d_ring_artifact_removal, estimate_noise_sigma, fft2d,
    multiscale_bm3d_streak_removal, run_bm3d_kernel, run_bm3d_step, run_bm3d_step_stack,
};
use ndarray::{Array, Array1, Array2, Array3, ArrayView2, Axis, Dimension, ShapeBuilder, s};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

const KERNEL_MODES: [Bm3dMode; 2] = [Bm3dMode::HardThreshold, Bm3dMode::Wiener];
const RING_MODES: [RingRemovalMode; 3] = [
    RingRemovalMode::Generic,
    RingRemovalMode::Streak,
    RingRemovalMode::FourierSvd,
];
// Not square, so a transposed read cannot line up with the true layout.
const ROWS: usize = 40;
const COLS: usize = 56;

/// Memory layouts of a 2D array that hold the same values.
#[derive(Debug, Clone, Copy)]
enum Layout2 {
    /// Column-major (Fortran order), as the transpose of a standard array.
    ColumnMajor,
    /// Negative column stride: the reversed view of a reversed copy.
    ReversedColumns,
    ReversedRows,
    /// Every second column of a wider array (not contiguous).
    EverySecondColumn,
}

const LAYOUTS_2D: [Layout2; 4] = [
    Layout2::ColumnMajor,
    Layout2::ReversedColumns,
    Layout2::ReversedRows,
    Layout2::EverySecondColumn,
];

/// Memory layouts of a 3D stack that hold the same values.
#[derive(Debug, Clone, Copy)]
enum Layout3 {
    /// Column-major stack. With one slice the slice is column-major too;
    /// with more, the slices are not contiguous.
    ColumnMajor,
    ColumnMajorSlices,
    ReversedColumns,
    /// Every second column of a wider stack (not contiguous).
    EverySecondColumn,
}

const LAYOUTS_3D: [Layout3; 4] = [
    Layout3::ColumnMajor,
    Layout3::ColumnMajorSlices,
    Layout3::ReversedColumns,
    Layout3::EverySecondColumn,
];

fn in_layout_2d<F: Bm3dFloat>(values: &Array2<F>, layout: Layout2) -> Array2<F> {
    in_layout(values, layout, F::nan())
}

/// `in_layout_2d` for any element type. Memory outside the copy holds
/// `filler`.
fn in_layout<T: Copy + PartialEq + Debug>(
    values: &Array2<T>,
    layout: Layout2,
    filler: T,
) -> Array2<T> {
    let (rows, cols) = values.dim();
    let out = match layout {
        Layout2::ColumnMajor => {
            let mut a = Array2::from_elem((rows, cols).f(), filler);
            a.assign(values);
            a
        }
        Layout2::ReversedColumns => {
            let mut a = values
                .slice(s![.., ..;-1])
                .as_standard_layout()
                .into_owned();
            a.invert_axis(Axis(1));
            a
        }
        Layout2::ReversedRows => {
            let mut a = values
                .slice(s![..;-1, ..])
                .as_standard_layout()
                .into_owned();
            a.invert_axis(Axis(0));
            a
        }
        Layout2::EverySecondColumn => {
            // The skipped columns hold `filler` (NaN in these tests), so
            // reading them shows in the output.
            let mut wide = Array2::from_elem((rows, 2 * cols), filler);
            wide.slice_mut(s![.., ..;2]).assign(values);
            wide.slice_collapse(s![.., ..;2]);
            wide
        }
    };
    assert_eq!(out, *values, "{layout:?} copy changed the values");
    assert!(
        !out.is_standard_layout(),
        "{layout:?} copy is in standard layout"
    );
    out
}

fn in_layout_3d<F: Bm3dFloat>(values: &Array3<F>, layout: Layout3) -> Array3<F> {
    let (n, rows, cols) = values.dim();
    let out = match layout {
        Layout3::ColumnMajor => {
            let mut a = Array3::zeros((n, rows, cols).f());
            a.assign(values);
            a
        }
        Layout3::ColumnMajorSlices => {
            let mut a = Array3::zeros((n, cols, rows));
            a.assign(&values.view().permuted_axes([0, 2, 1]));
            let a = a.permuted_axes([0, 2, 1]);
            assert!(a.index_axis(Axis(0), 0).t().is_standard_layout());
            a
        }
        Layout3::ReversedColumns => {
            let mut a = values
                .slice(s![.., .., ..;-1])
                .as_standard_layout()
                .into_owned();
            a.invert_axis(Axis(2));
            a
        }
        Layout3::EverySecondColumn => {
            let mut wide = Array3::from_elem((n, rows, 2 * cols), F::nan());
            wide.slice_mut(s![.., .., ..;2]).assign(values);
            wide.slice_collapse(s![.., .., ..;2]);
            wide
        }
    };
    assert_eq!(out, *values, "{layout:?} copy changed the values");
    assert!(
        !out.index_axis(Axis(0), 0).is_standard_layout(),
        "{layout:?} copy has standard-layout slices"
    );
    out
}

fn noisy_image<F: Bm3dFloat>(rows: usize, cols: usize, seed: u64) -> Array2<F> {
    let mut state = seed;
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = (state >> 40) as f64 / (1u64 << 24) as f64 - 0.5;
        let truth = if c >= 29 { 0.75 } else { 0.3 } + if r >= 19 { 0.1 } else { 0.0 };
        F::from_f64_c(truth + 0.002 * c as f64 + 0.15 * noise)
    })
}

fn noisy_stack<F: Bm3dFloat>(slices: usize) -> Array3<F> {
    let mut stack = Array3::zeros((slices, ROWS, COLS));
    for (i, mut slice) in stack.outer_iter_mut().enumerate() {
        slice.assign(&noisy_image::<F>(ROWS, COLS, 31 + i as u64));
    }
    stack
}

fn streak_psd<F: Bm3dFloat>(patch_size: usize) -> Array2<F> {
    Array2::from_shape_fn((patch_size, patch_size), |(r, _)| {
        let d = r.min(patch_size - r) as f64 / 0.6;
        F::from_f64_c((-0.5 * d * d).exp())
    })
}

fn sigma_map<F: Bm3dFloat>(rows: usize, cols: usize) -> Array2<F> {
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        F::from_f64_c(0.01 + 0.02 * ((c % 7) as f64 / 7.0) + 0.001 * r as f64)
    })
}

fn tolerance<F: Bm3dFloat>() -> f64 {
    if std::mem::size_of::<F>() == std::mem::size_of::<f32>() {
        1e-3
    } else {
        1e-12
    }
}

fn max_abs_diff<F: Bm3dFloat, D: Dimension>(a: &Array<F, D>, b: &Array<F, D>) -> f64 {
    assert_eq!(a.shape(), b.shape());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y).abs().to_f64().unwrap())
        .fold(0.0, |m, d| if d.is_nan() || d > m { d } else { m })
}

fn assert_close<F: Bm3dFloat, D: Dimension>(label: &str, got: &Array<F, D>, want: &Array<F, D>) {
    let diff = max_abs_diff(got, want);
    let scale = want
        .iter()
        .map(|v| v.abs().to_f64().unwrap())
        .fold(0.0, f64::max);
    let bound = tolerance::<F>() * scale;
    assert!(
        diff <= bound,
        "{label}: max |non-standard - standard| = {diff:e}, bound {bound:e} (max |standard| {scale:e})"
    );
}

fn kernel_config<F: Bm3dFloat>(patch_size: usize) -> Bm3dKernelConfig<F> {
    Bm3dKernelConfig {
        sigma_random: F::from_f64_c(0.05),
        threshold: F::from_f64_c(2.7),
        patch_size,
        step_size: 3,
        search_window: 16,
        max_matches: 16,
        search_step: 1,
        use_hadamard_fast_path: None,
    }
}

struct KernelInputs<F: Bm3dFloat> {
    noisy: Array2<F>,
    pilot: Array2<F>,
    psd: Array2<F>,
    map: Array2<F>,
}

fn kernel_inputs<F: Bm3dFloat>(mode: Bm3dMode, patch_size: usize) -> KernelInputs<F> {
    let noisy = noisy_image::<F>(ROWS, COLS, 7);
    let pilot = match mode {
        Bm3dMode::HardThreshold => noisy.clone(),
        // The Wiener pass takes a pilot estimate; another image will do.
        Bm3dMode::Wiener => noisy_image::<F>(ROWS, COLS, 8),
    };
    KernelInputs {
        noisy,
        pilot,
        psd: streak_psd::<F>(patch_size),
        map: sigma_map::<F>(ROWS, COLS),
    }
}

/// A 2D kernel entry point, `run_bm3d_step` or `run_bm3d_kernel`.
type Kernel<F> = fn(&KernelInputs<F>, Bm3dMode, &Bm3dKernelConfig<F>, &Bm3dPlans<F>) -> Array2<F>;

fn run_step<F: Bm3dFloat>(
    inputs: &KernelInputs<F>,
    mode: Bm3dMode,
    config: &Bm3dKernelConfig<F>,
    plans: &Bm3dPlans<F>,
) -> Array2<F> {
    run_bm3d_step(
        inputs.noisy.view(),
        inputs.pilot.view(),
        mode,
        inputs.psd.view(),
        inputs.map.view(),
        config,
        plans,
    )
    .expect("run_bm3d_step failed")
}

fn run_kernel<F: Bm3dFloat>(
    inputs: &KernelInputs<F>,
    mode: Bm3dMode,
    config: &Bm3dKernelConfig<F>,
    plans: &Bm3dPlans<F>,
) -> Array2<F> {
    run_bm3d_kernel(
        inputs.noisy.view(),
        inputs.pilot.view(),
        mode,
        inputs.psd.view(),
        inputs.map.view(),
        config,
        plans,
    )
}

fn check_kernel_entry<F: Bm3dFloat>(name: &str, precision: &str, kernel: Kernel<F>) {
    for patch_size in [8, 7] {
        let config = kernel_config::<F>(patch_size);
        let plans = Bm3dPlans::new(patch_size, config.max_matches);
        for mode in KERNEL_MODES {
            let standard = kernel_inputs::<F>(mode, patch_size);
            let want = kernel(&standard, mode, &config, &plans);
            for layout in LAYOUTS_2D {
                let inputs = KernelInputs {
                    noisy: in_layout_2d(&standard.noisy, layout),
                    pilot: in_layout_2d(&standard.pilot, layout),
                    psd: in_layout_2d(&standard.psd, layout),
                    map: in_layout_2d(&standard.map, layout),
                };
                let got = kernel(&inputs, mode, &config, &plans);
                assert_close(
                    &format!("{name} {precision} {mode:?} patch {patch_size} {layout:?}"),
                    &got,
                    &want,
                );
            }
        }
    }
}

#[test]
fn run_bm3d_step_gives_the_same_result_for_any_layout() {
    check_kernel_entry::<f32>("run_bm3d_step", "f32", run_step);
    check_kernel_entry::<f64>("run_bm3d_step", "f64", run_step);
}

#[test]
fn run_bm3d_kernel_gives_the_same_result_for_any_layout() {
    check_kernel_entry::<f32>("run_bm3d_kernel", "f32", run_kernel);
    check_kernel_entry::<f64>("run_bm3d_kernel", "f64", run_kernel);
}

fn run_stack<F: Bm3dFloat>(
    noisy: &Array3<F>,
    pilot: &Array3<F>,
    map: &Array3<F>,
    psd: &Array2<F>,
    mode: Bm3dMode,
    config: &Bm3dKernelConfig<F>,
    plans: &Bm3dPlans<F>,
) -> Array3<F> {
    run_bm3d_step_stack(
        noisy.view(),
        pilot.view(),
        mode,
        psd.view(),
        map.view(),
        config,
        plans,
        None,
    )
    .expect("run_bm3d_step_stack failed")
}

fn check_run_bm3d_step_stack<F: Bm3dFloat>(precision: &str) {
    let patch_size = 8;
    let config = kernel_config::<F>(patch_size);
    let plans = Bm3dPlans::new(patch_size, config.max_matches);
    let psd = streak_psd::<F>(patch_size);
    let map_2d = sigma_map::<F>(ROWS, COLS);
    for slices in [1, 2] {
        let noisy = noisy_stack::<F>(slices);
        let map = Array3::from_shape_fn((slices, ROWS, COLS), |(i, r, c)| {
            map_2d[[r, c]] + F::from_f64_c(0.003 * i as f64)
        });
        for mode in KERNEL_MODES {
            let pilot = match mode {
                Bm3dMode::HardThreshold => noisy.clone(),
                Bm3dMode::Wiener => noisy.mapv(|v| v * F::from_f64_c(0.97)),
            };
            let want = run_stack(&noisy, &pilot, &map, &psd, mode, &config, &plans);
            for layout in LAYOUTS_3D {
                let got = run_stack(
                    &in_layout_3d(&noisy, layout),
                    &in_layout_3d(&pilot, layout),
                    &in_layout_3d(&map, layout),
                    &psd,
                    mode,
                    &config,
                    &plans,
                );
                assert_close(
                    &format!("run_bm3d_step_stack {precision} {mode:?} {slices} slices {layout:?}"),
                    &got,
                    &want,
                );
            }
        }
    }
}

#[test]
fn run_bm3d_step_stack_gives_the_same_result_for_any_layout() {
    check_run_bm3d_step_stack::<f32>("f32");
    check_run_bm3d_step_stack::<f64>("f64");
}

fn check_ring_removal<F: Bm3dFloat>(precision: &str) {
    let image = noisy_image::<F>(ROWS, COLS, 9);
    for sigma_random in [0.08, 0.0] {
        // sigma_random 0 also runs the noise estimator on the input layout.
        let config = Bm3dConfig::<F> {
            sigma_random: F::from_f64_c(sigma_random),
            ..Bm3dConfig::default()
        };
        for mode in RING_MODES {
            let want = bm3d_ring_artifact_removal(image.view(), mode, &config).unwrap();
            for layout in LAYOUTS_2D {
                let input = in_layout_2d(&image, layout);
                let got = bm3d_ring_artifact_removal(input.view(), mode, &config).unwrap();
                assert_close(
                    &format!(
                        "bm3d_ring_artifact_removal {precision} {mode:?} sigma {sigma_random} {layout:?}"
                    ),
                    &got,
                    &want,
                );
            }
        }
    }
}

#[test]
fn ring_removal_gives_the_same_result_for_any_layout() {
    check_ring_removal::<f32>("f32");
    check_ring_removal::<f64>("f64");
}

fn check_multiscale<F: Bm3dFloat>(precision: &str) {
    // Width 56 gives no binned scale (single-scale streak BM3D on the input
    // itself); width 96 gives one.
    for cols in [56, 96] {
        let image = noisy_image::<F>(ROWS, cols, 13);
        for log_domain_input in [false, true] {
            let config = MultiscaleConfig::<F> {
                log_domain_input,
                ..MultiscaleConfig::default()
            };
            let want = multiscale_bm3d_streak_removal(image.view(), &config).unwrap();
            for layout in LAYOUTS_2D {
                let input = in_layout_2d(&image, layout);
                let got = multiscale_bm3d_streak_removal(input.view(), &config).unwrap();
                assert_close(
                    &format!(
                        "multiscale {precision} width {cols} log_domain_input={log_domain_input} {layout:?}"
                    ),
                    &got,
                    &want,
                );
            }
        }
    }
}

#[test]
fn multiscale_gives_the_same_result_for_any_layout() {
    check_multiscale::<f32>("f32");
    check_multiscale::<f64>("f64");
}

fn check_noise_estimate<F: Bm3dFloat>(precision: &str) {
    let image = noisy_image::<F>(ROWS, COLS, 17);
    let want = Array::from_elem((), estimate_noise_sigma(image.view()));
    for layout in LAYOUTS_2D {
        let input = in_layout_2d(&image, layout);
        let got = Array::from_elem((), estimate_noise_sigma(input.view()));
        assert_close(
            &format!("estimate_noise_sigma {precision} {layout:?}"),
            &got,
            &want,
        );
    }
}

#[test]
fn noise_estimate_is_the_same_for_any_layout() {
    check_noise_estimate::<f32>("f32");
    check_noise_estimate::<f64>("f64");
}

/// Real and imaginary parts side by side, for one tolerance check.
fn complex_parts<F: Bm3dFloat>(spectrum: &Array2<Complex<F>>) -> Array2<F> {
    let (rows, cols) = spectrum.dim();
    let mut parts = Array2::zeros((rows, 2 * cols));
    for ((r, c), z) in spectrum.indexed_iter() {
        parts[[r, 2 * c]] = z.re;
        parts[[r, 2 * c + 1]] = z.im;
    }
    parts
}

fn spectrum_of<F: Bm3dFloat>(input: ArrayView2<F>) -> Array2<Complex<F>> {
    let (rows, cols) = input.dim();
    let mut planner = FftPlanner::<F>::new();
    let row_plan = planner.plan_fft_forward(cols);
    let col_plan = planner.plan_fft_forward(rows);
    fft2d(input, &row_plan, &col_plan)
}

fn fft_of<F: Bm3dFloat>(input: ArrayView2<F>) -> Array2<F> {
    complex_parts(&spectrum_of(input))
}

fn check_fft2d<F: Bm3dFloat>(precision: &str) {
    let patch = noisy_image::<F>(ROWS, COLS, 19)
        .slice(s![3..11, 25..33])
        .to_owned();
    let want = fft_of(patch.view());
    for layout in LAYOUTS_2D {
        let input = in_layout_2d(&patch, layout);
        let got = fft_of(input.view());
        assert_close(&format!("fft2d {precision} {layout:?}"), &got, &want);
    }
}

#[test]
fn fft2d_gives_the_same_result_for_any_layout() {
    check_fft2d::<f32>("f32");
    check_fft2d::<f64>("f64");
}

/// A square patch (in-place transpose branch) and a non-square patch (column
/// loop branch) for the 2D transforms.
const TRANSFORM_SHAPES: [(usize, usize); 2] = [(8, 8), (8, 12)];

/// A patch across the step edge at column 29 of a noisy image.
fn transform_patch<F: Bm3dFloat>(rows: usize, cols: usize) -> Array2<F> {
    noisy_image::<F>(ROWS, COLS, 19)
        .slice(s![3..3 + rows, 23..23 + cols])
        .to_owned()
}

fn complex_nan<F: Bm3dFloat>() -> Complex<F> {
    Complex::new(F::nan(), F::nan())
}

fn complex_zeros<F: Bm3dFloat>(rows: usize, cols: usize) -> Array2<Complex<F>> {
    Array2::from_elem((rows, cols), Complex::new(F::zero(), F::zero()))
}

/// `fft2d_into` of `input` into `output`, which the call overwrites.
fn fft_into<F: Bm3dFloat>(
    input: ArrayView2<F>,
    mut output: Array2<Complex<F>>,
) -> Array2<Complex<F>> {
    let (rows, cols) = input.dim();
    let mut planner = FftPlanner::<F>::new();
    let row_plan = planner.plan_fft_forward(cols);
    let col_plan = planner.plan_fft_forward(rows);
    let mut work = complex_zeros(rows, cols);
    let mut scratch = vec![Complex::new(F::zero(), F::zero()); rows.max(cols)];
    fft2d_into(
        input,
        &row_plan,
        &col_plan,
        &mut work,
        output.view_mut(),
        &mut scratch,
    );
    output
}

/// `ifft2d_into` of `spectrum` into `output`, which the call overwrites.
fn ifft_into<F: Bm3dFloat>(spectrum: ArrayView2<Complex<F>>, mut output: Array2<F>) -> Array2<F> {
    let (rows, cols) = spectrum.dim();
    let mut planner = FftPlanner::<F>::new();
    let row_plan = planner.plan_fft_inverse(cols);
    let col_plan = planner.plan_fft_inverse(rows);
    let mut work = complex_zeros(rows, cols);
    let mut scratch = vec![Complex::new(F::zero(), F::zero()); rows.max(cols)];
    ifft2d_into(
        spectrum,
        &row_plan,
        &col_plan,
        &mut work,
        &mut output,
        &mut scratch,
    );
    output
}

/// `ifft2d_inplace_to_real_with_plan_scratch` of `spectrum` into `output`.
/// The call overwrites both.
fn ifft_in_place<F: Bm3dFloat>(
    mut spectrum: Array2<Complex<F>>,
    mut output: Array2<F>,
) -> Array2<F> {
    let (rows, cols) = spectrum.dim();
    let mut planner = FftPlanner::<F>::new();
    let row_plan = planner.plan_fft_inverse(cols);
    let col_plan = planner.plan_fft_inverse(rows);
    let zero = Complex::new(F::zero(), F::zero());
    let mut scratch = vec![zero; rows.max(cols)];
    let mut row_fft_scratch = vec![zero; row_plan.get_inplace_scratch_len()];
    let mut col_fft_scratch = vec![zero; col_plan.get_inplace_scratch_len()];
    ifft2d_inplace_to_real_with_plan_scratch(
        spectrum.view_mut(),
        &row_plan,
        &col_plan,
        &mut output,
        &mut scratch,
        &mut row_fft_scratch,
        &mut col_fft_scratch,
    );
    output
}

fn check_fft2d_into_output<F: Bm3dFloat>(precision: &str) {
    for (rows, cols) in TRANSFORM_SHAPES {
        let patch = transform_patch::<F>(rows, cols);
        let want = complex_parts(&fft_into(patch.view(), complex_zeros(rows, cols)));
        for layout in LAYOUTS_2D {
            let output = in_layout(&complex_zeros(rows, cols), layout, complex_nan());
            let got = complex_parts(&fft_into(patch.view(), output));
            assert_close(
                &format!("fft2d_into {precision} {rows}x{cols} output {layout:?}"),
                &got,
                &want,
            );
        }
    }
}

#[test]
fn fft2d_into_writes_an_output_buffer_in_any_layout() {
    check_fft2d_into_output::<f32>("f32");
    check_fft2d_into_output::<f64>("f64");
}

fn check_ifft2d_into_input<F: Bm3dFloat>(precision: &str) {
    for (rows, cols) in TRANSFORM_SHAPES {
        let spectrum = spectrum_of(transform_patch::<F>(rows, cols).view());
        let want = ifft_into(spectrum.view(), Array2::zeros((rows, cols)));
        for layout in LAYOUTS_2D {
            let input = in_layout(&spectrum, layout, complex_nan());
            let got = ifft_into(input.view(), Array2::zeros((rows, cols)));
            assert_close(
                &format!("ifft2d_into {precision} {rows}x{cols} input {layout:?}"),
                &got,
                &want,
            );
        }
    }
}

#[test]
fn ifft2d_into_reads_an_input_in_any_layout() {
    check_ifft2d_into_input::<f32>("f32");
    check_ifft2d_into_input::<f64>("f64");
}

fn check_ifft2d_into_output<F: Bm3dFloat>(precision: &str) {
    for (rows, cols) in TRANSFORM_SHAPES {
        let spectrum = spectrum_of(transform_patch::<F>(rows, cols).view());
        let want = ifft_into(spectrum.view(), Array2::zeros((rows, cols)));
        for layout in LAYOUTS_2D {
            let output = in_layout_2d(&Array2::zeros((rows, cols)), layout);
            let got = ifft_into(spectrum.view(), output);
            assert_close(
                &format!("ifft2d_into {precision} {rows}x{cols} output {layout:?}"),
                &got,
                &want,
            );
        }
    }
}

#[test]
fn ifft2d_into_writes_an_output_buffer_in_any_layout() {
    check_ifft2d_into_output::<f32>("f32");
    check_ifft2d_into_output::<f64>("f64");
}

fn check_ifft2d_in_place_input<F: Bm3dFloat>(precision: &str) {
    for (rows, cols) in TRANSFORM_SHAPES {
        let spectrum = spectrum_of(transform_patch::<F>(rows, cols).view());
        let want = ifft_in_place(spectrum.clone(), Array2::zeros((rows, cols)));
        for layout in LAYOUTS_2D {
            let input_output = in_layout(&spectrum, layout, complex_nan());
            let got = ifft_in_place(input_output, Array2::zeros((rows, cols)));
            assert_close(
                &format!("ifft2d_inplace_to_real {precision} {rows}x{cols} input {layout:?}"),
                &got,
                &want,
            );
        }
    }
}

#[test]
fn ifft2d_in_place_reads_an_input_in_any_layout() {
    check_ifft2d_in_place_input::<f32>("f32");
    check_ifft2d_in_place_input::<f64>("f64");
}

fn check_ifft2d_in_place_output<F: Bm3dFloat>(precision: &str) {
    for (rows, cols) in TRANSFORM_SHAPES {
        let spectrum = spectrum_of(transform_patch::<F>(rows, cols).view());
        let want = ifft_in_place(spectrum.clone(), Array2::zeros((rows, cols)));
        for layout in LAYOUTS_2D {
            let output = in_layout_2d(&Array2::zeros((rows, cols)), layout);
            let got = ifft_in_place(spectrum.clone(), output);
            assert_close(
                &format!("ifft2d_inplace_to_real {precision} {rows}x{cols} output {layout:?}"),
                &got,
                &want,
            );
        }
    }
}

#[test]
fn ifft2d_in_place_writes_an_output_buffer_in_any_layout() {
    check_ifft2d_in_place_output::<f32>("f32");
    check_ifft2d_in_place_output::<f64>("f64");
}

/// Reference positions: inside, top-left corner, bottom-right corner.
const MATCH_REFS: [(usize, usize); 3] = [(16, 24), (0, 0), (32, 48)];
/// Standard layout (None) or column-major, for the argument a test holds.
const STANDARD_OR_COLUMN_MAJOR: [Option<Layout2>; 2] = [None, Some(Layout2::ColumnMajor)];

fn matches_of<F: Bm3dFloat>(
    image: ArrayView2<F>,
    integral_sum: &Array2<F>,
    ref_pos: (usize, usize),
    patch_size: usize,
) -> Vec<PatchMatch<F>> {
    let mut matches = Vec::new();
    find_similar_patches_in_place_sum(
        image,
        integral_sum,
        ref_pos,
        (patch_size, patch_size),
        (16, 16),
        16,
        1,
        &mut matches,
    );
    matches
}

fn assert_same_matches<F: Bm3dFloat>(label: &str, got: &[PatchMatch<F>], want: &[PatchMatch<F>]) {
    let by_position = |matches: &[PatchMatch<F>]| {
        let mut sorted = matches.to_vec();
        sorted.sort_by_key(|m| (m.row, m.col));
        sorted
    };
    let (got, want) = (by_position(got), by_position(want));
    let positions = |v: &[PatchMatch<F>]| v.iter().map(|m| (m.row, m.col)).collect::<Vec<_>>();
    assert_eq!(
        positions(&got),
        positions(&want),
        "{label}: different matches"
    );
    let distances = |v: &[PatchMatch<F>]| Array1::from_iter(v.iter().map(|m| m.distance));
    assert_close(label, &distances(&got), &distances(&want));
}

/// Each case is (image layout, integral-sum layout); None is standard layout.
fn check_find_similar_patches<F: Bm3dFloat>(
    precision: &str,
    cases: &[(Option<Layout2>, Option<Layout2>)],
) {
    let image = noisy_image::<F>(ROWS, COLS, 23);
    let integral_sum = compute_integral_sum_image(image.view());
    let copy = |a: &Array2<F>, layout: Option<Layout2>| match layout {
        None => a.clone(),
        Some(layout) => in_layout_2d(a, layout),
    };
    for patch_size in [8, 7] {
        for ref_pos in MATCH_REFS {
            let want = matches_of(image.view(), &integral_sum, ref_pos, patch_size);
            for &(image_layout, integral_layout) in cases {
                let got = matches_of(
                    copy(&image, image_layout).view(),
                    &copy(&integral_sum, integral_layout),
                    ref_pos,
                    patch_size,
                );
                assert_same_matches(
                    &format!(
                        "find_similar_patches_in_place_sum {precision} patch {patch_size} \
                         ref {ref_pos:?} image {image_layout:?} integral_sum {integral_layout:?}"
                    ),
                    &got,
                    &want,
                );
            }
        }
    }
}

#[test]
fn find_similar_patches_reads_an_integral_sum_in_any_layout() {
    let cases: Vec<_> = STANDARD_OR_COLUMN_MAJOR
        .iter()
        .flat_map(|&image| LAYOUTS_2D.map(|layout| (image, Some(layout))))
        .collect();
    check_find_similar_patches::<f32>("f32", &cases);
    check_find_similar_patches::<f64>("f64", &cases);
}

#[test]
fn find_similar_patches_reads_an_image_in_any_layout() {
    let cases: Vec<_> = STANDARD_OR_COLUMN_MAJOR
        .iter()
        .flat_map(|&integral_sum| LAYOUTS_2D.map(|layout| (Some(layout), integral_sum)))
        .collect();
    check_find_similar_patches::<f32>("f32", &cases);
    check_find_similar_patches::<f64>("f64", &cases);
}
