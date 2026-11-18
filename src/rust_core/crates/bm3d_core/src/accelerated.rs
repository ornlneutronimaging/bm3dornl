//! BM3D passes through heterocompute, which runs them on the CPU, a CUDA GPU or an Apple GPU,
//! wherever its planner predicts them fastest from rates it measures once per machine.
//!
//! heterocompute's filter is this crate's kernel with every operation's order fixed, so every
//! device gives the same bits; its results agree with this crate's own to rounding, not bit for
//! bit. `BM3DORNL_DEVICE` set to `cpu` or `gpu` forces a device; `BM3D_PROFILE_TIMING` prints
//! each call's report. A call heterocompute refuses runs on this crate's own kernel.

use crate::float_trait::Bm3dFloat;
use crate::pipeline::Bm3dKernelConfig;
use heterocompute::{
    Accuracy, Bm3dNoise, Bm3dParams, Bm3dPasses, Bm3dTransform, CpuElement, Engine, HostMatrix,
    HostMatrixMut, Placement, Request, SigmaMap,
};
use ndarray::{Array3, ArrayView2, ArrayView3};
use std::any::TypeId;
use std::sync::OnceLock;

/// The engine every call shares, so the devices and rates are found once.
fn engine() -> &'static Engine {
    static ENGINE: OnceLock<Engine> = OnceLock::new();
    ENGINE.get_or_init(Engine::new)
}

/// The passes a call runs.
#[derive(Clone, Copy)]
pub(crate) enum Passes<'a, F> {
    /// Hard thresholding, the noisy images their own pilot.
    Hard,
    /// Wiener filtering on these pilot images.
    Wiener(ArrayView3<'a, F>),
    /// Hard thresholding, then Wiener filtering on its result.
    Both,
}

/// The request for `T`: f32 accuracy for f32, on the device `BM3DORNL_DEVICE` names.
fn request<T: CpuElement>() -> Request {
    let placement = match std::env::var("BM3DORNL_DEVICE").as_deref() {
        Ok("cpu") => Placement::Cpu,
        Ok("gpu") => Placement::Gpu,
        _ => Placement::Auto,
    };
    let accuracy = if size_of::<T>() == 4 {
        Accuracy::F32
    } else {
        Accuracy::F64
    };
    Request::default().accuracy(accuracy).on(placement)
}

/// The passes of `noisy`, `(n, rows, cols)`, with the PSD `sigma_psd` (`p x p`, else white)
/// and the sigma map `sigma_map` (`(n, rows, cols)`, `(n, 1, cols)`, else none), as this crate's
/// kernel runs them with `config`; `None` where heterocompute refuses the call.
pub(crate) fn run<F: Bm3dFloat>(
    noisy: ArrayView3<F>,
    passes: Passes<F>,
    sigma_psd: ArrayView2<F>,
    sigma_map: ArrayView3<F>,
    config: &Bm3dKernelConfig<F>,
    hadamard: bool,
) -> Option<Array3<F>> {
    if TypeId::of::<F>() == TypeId::of::<f32>() {
        // SAFETY: `F` is `f32`, so the views and arrays are of `f32`.
        unsafe { typed::<F, f32>(noisy, passes, sigma_psd, sigma_map, config, hadamard) }
    } else if TypeId::of::<F>() == TypeId::of::<f64>() {
        // SAFETY: `F` is `f64`, so the views and arrays are of `f64`.
        unsafe { typed::<F, f64>(noisy, passes, sigma_psd, sigma_map, config, hadamard) }
    } else {
        None
    }
}

/// [`run`] on `T`, the type `F` is.
///
/// # Safety
/// `F` and `T` are the same type.
unsafe fn typed<F: Bm3dFloat, T: CpuElement + Copy>(
    noisy: ArrayView3<F>,
    passes: Passes<F>,
    sigma_psd: ArrayView2<F>,
    sigma_map: ArrayView3<F>,
    config: &Bm3dKernelConfig<F>,
    hadamard: bool,
) -> Option<Array3<F>> {
    // SAFETY: the caller's contract: an `F` is a `T`.
    let cast = |v: F| unsafe { std::mem::transmute_copy::<F, T>(&v) };
    let back = |v: T| unsafe { std::mem::transmute_copy::<T, F>(&v) };
    let (n, rows, cols) = noisy.dim();
    let flat = |a: ArrayView3<F>| -> Vec<T> { a.iter().map(|&v| cast(v)).collect() };
    let images = flat(noisy);
    let pilot = match passes {
        Passes::Wiener(pilot) => Some(flat(pilot)),
        _ => None,
    };
    let p = config.patch_size;
    let psd: Option<Vec<T>> =
        (sigma_psd.dim() == (p, p)).then(|| sigma_psd.iter().map(|&v| cast(v)).collect());
    let map = flat(sigma_map);
    let noisy_view = HostMatrix::new(&images, n, rows * cols).ok()?;
    let sigma = match sigma_map.dim() {
        d if d == (n, rows, cols) => SigmaMap::Full(HostMatrix::new(&map, n, rows * cols).ok()?),
        d if d == (n, 1, cols) => SigmaMap::Columns(HostMatrix::new(&map, n, cols).ok()?),
        _ => SigmaMap::Zero,
    };
    let noise = Bm3dNoise {
        psd: psd.as_deref(),
        sigma,
    };
    let params = Bm3dParams {
        patch: p,
        step: config.step_size,
        search_window: config.search_window,
        search_step: config.search_step,
        max_matches: config.max_matches,
        threshold: config.threshold.to_f64().unwrap_or(f64::NAN),
        sigma_random: config.sigma_random.to_f64().unwrap_or(f64::NAN),
        transform: if hadamard {
            Bm3dTransform::Hadamard
        } else {
            Bm3dTransform::Fourier
        },
    };
    let passes = match (passes, &pilot) {
        (Passes::Hard, _) => Bm3dPasses::Hard,
        (Passes::Both, _) => Bm3dPasses::Both,
        (Passes::Wiener(_), Some(pilot)) => {
            Bm3dPasses::Wiener(HostMatrix::new(pilot, n, rows * cols).ok()?)
        }
        (Passes::Wiener(_), None) => return None,
    };
    let mut out = vec![T::default(); n * rows * cols];
    let mut out_view = HostMatrixMut::new(&mut out, n, rows * cols).ok()?;
    let image = (rows, cols);
    let request = request::<T>();
    let ran = engine().bm3d(
        noisy_view,
        image,
        passes,
        &params,
        &noise,
        &mut out_view,
        request,
    );
    let report = ran.ok()?;
    if std::env::var_os("BM3D_PROFILE_TIMING").is_some() {
        eprint!("{}", report.explain());
    }
    let values: Vec<F> = out.into_iter().map(back).collect();
    Array3::from_shape_vec((n, rows, cols), values).ok()
}
