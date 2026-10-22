//! `Bm3dConfig::search_step`, the candidate-search stride.
//!
//! `step_size` spaces the reference patches; `search_step` spaces the
//! candidate positions that block matching tests around each reference.
//! `None` resolves per mode: every position (1) in generic mode and
//! `step_size` in streak mode, so also in multiscale mode, which runs streak
//! mode at each scale. Fourier-SVD mode also keeps `step_size`. An explicit
//! value is used as given.

use bm3d_core::{
    Bm3dConfig, Bm3dFloat, MultiscaleConfig, RingRemovalMode, bm3d_ring_artifact_removal,
    multiscale_bm3d_streak_removal,
};
use ndarray::Array2;

/// Step edges off the step-4 grid (column 41, row 35) plus uniform noise
/// from a fixed-seed linear congruential generator.
fn noisy_step_edges<F: Bm3dFloat>(rows: usize, cols: usize, seed: u64) -> Array2<F> {
    let mut state = seed;
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = (state >> 40) as f64 / (1u64 << 24) as f64 - 0.5;
        let truth = if c >= 41 { 0.75 } else { 0.25 } + if r >= 35 { 0.15 } else { 0.0 };
        F::from_f64_c(truth + 0.2 * noise)
    })
}

fn single_scale<F: Bm3dFloat>(
    mode: RingRemovalMode,
    step_size: usize,
    search_step: Option<usize>,
) -> Array2<F> {
    let image = noisy_step_edges::<F>(64, 80, 7);
    let config = Bm3dConfig {
        sigma_random: F::from_f64_c(0.07),
        step_size,
        search_step,
        ..Bm3dConfig::default()
    };
    bm3d_ring_artifact_removal(image.view(), mode, &config).unwrap()
}

fn multiscale<F: Bm3dFloat>(search_step: Option<usize>) -> Array2<F> {
    // Width 160 gives two binned scales.
    let image = noisy_step_edges::<F>(64, 160, 11);
    let mut config = MultiscaleConfig::<F>::default();
    config.bm3d_config.search_step = search_step;
    multiscale_bm3d_streak_removal(image.view(), &config).unwrap()
}

#[test]
fn default_search_step_is_none() {
    assert_eq!(Bm3dConfig::<f32>::default().search_step, None);
    assert_eq!(Bm3dConfig::<f64>::default().search_step, None);
    assert_eq!(
        MultiscaleConfig::<f32>::default().bm3d_config.search_step,
        None
    );
    assert_eq!(
        MultiscaleConfig::<f64>::default().bm3d_config.search_step,
        None
    );
}

#[test]
fn validate_rejects_zero_search_step() {
    let zero = Bm3dConfig::<f32> {
        search_step: Some(0),
        ..Bm3dConfig::default()
    };
    let err = zero
        .validate()
        .expect_err("search_step = Some(0) must fail validation");
    assert!(
        err.contains("search_step"),
        "the error should name search_step, got: {err}"
    );
    for search_step in [None, Some(1), Some(2), Some(4), Some(7)] {
        let config = Bm3dConfig::<f32> {
            search_step,
            ..Bm3dConfig::default()
        };
        assert!(
            config.validate().is_ok(),
            "search_step = {search_step:?} should validate"
        );
    }
}

#[test]
fn public_entry_points_reject_zero_search_step() {
    let image = noisy_step_edges::<f32>(64, 160, 3);
    let config = Bm3dConfig::<f32> {
        search_step: Some(0),
        ..Bm3dConfig::default()
    };
    for mode in [RingRemovalMode::Generic, RingRemovalMode::Streak] {
        assert!(
            bm3d_ring_artifact_removal(image.view(), mode, &config).is_err(),
            "{mode:?} mode accepted search_step = Some(0)"
        );
    }
    let mut multiscale_config = MultiscaleConfig::<f32>::default();
    multiscale_config.bm3d_config.search_step = Some(0);
    assert!(
        multiscale_bm3d_streak_removal(image.view(), &multiscale_config).is_err(),
        "multiscale accepted search_step = Some(0)"
    );
}

fn check_generic_default_searches_every_position<F: Bm3dFloat>() {
    let default = single_scale::<F>(RingRemovalMode::Generic, 4, None);
    assert!(
        default == single_scale::<F>(RingRemovalMode::Generic, 4, Some(1)),
        "generic default must equal search_step = Some(1)"
    );
    assert!(
        default != single_scale::<F>(RingRemovalMode::Generic, 4, Some(4)),
        "generic default must not search on the step_size grid"
    );
}

#[test]
fn generic_default_searches_every_position() {
    check_generic_default_searches_every_position::<f32>();
    check_generic_default_searches_every_position::<f64>();
}

fn check_streak_default_searches_on_step_size<F: Bm3dFloat>() {
    for step_size in [4, 2] {
        let default = single_scale::<F>(RingRemovalMode::Streak, step_size, None);
        assert!(
            default == single_scale::<F>(RingRemovalMode::Streak, step_size, Some(step_size)),
            "streak default must equal search_step = Some(step_size) at step_size {step_size}"
        );
        assert!(
            default != single_scale::<F>(RingRemovalMode::Streak, step_size, Some(1)),
            "streak default must not search every position at step_size {step_size}"
        );
    }
}

#[test]
fn streak_default_searches_on_step_size() {
    check_streak_default_searches_on_step_size::<f32>();
    check_streak_default_searches_on_step_size::<f64>();
}

fn check_fourier_svd_default_searches_on_step_size<F: Bm3dFloat>() {
    let default = single_scale::<F>(RingRemovalMode::FourierSvd, 4, None);
    assert!(
        default == single_scale::<F>(RingRemovalMode::FourierSvd, 4, Some(4)),
        "Fourier-SVD default must equal search_step = Some(step_size)"
    );
    assert!(
        default != single_scale::<F>(RingRemovalMode::FourierSvd, 4, Some(1)),
        "Fourier-SVD default must not search every position"
    );
}

#[test]
fn fourier_svd_default_searches_on_step_size() {
    check_fourier_svd_default_searches_on_step_size::<f32>();
    check_fourier_svd_default_searches_on_step_size::<f64>();
}

fn check_explicit_search_step_is_used<F: Bm3dFloat>() {
    for mode in [RingRemovalMode::Generic, RingRemovalMode::Streak] {
        let explicit = single_scale::<F>(mode, 4, Some(2));
        for other in [None, Some(1), Some(4)] {
            assert!(
                explicit != single_scale::<F>(mode, 4, other),
                "{mode:?}: search_step = Some(2) gave the same output as {other:?}"
            );
        }
    }
}

#[test]
fn explicit_search_step_is_used_in_every_mode() {
    check_explicit_search_step_is_used::<f32>();
    check_explicit_search_step_is_used::<f64>();
}

fn check_multiscale_default_searches_on_step_size<F: Bm3dFloat>() {
    let default = multiscale::<F>(None);
    assert!(
        default == multiscale::<F>(Some(4)),
        "multiscale default must equal search_step = Some(step_size)"
    );
    assert!(
        default != multiscale::<F>(Some(1)),
        "multiscale default must not search every position"
    );
}

#[test]
fn multiscale_default_searches_on_step_size() {
    check_multiscale_default_searches_on_step_size::<f32>();
    check_multiscale_default_searches_on_step_size::<f64>();
}
