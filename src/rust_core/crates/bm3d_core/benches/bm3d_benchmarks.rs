//! Criterion benchmarks for BM3D core operations.
//!
//! Run with: cargo bench -p bm3d_core
//! Run specific: cargo bench -p bm3d_core -- bench_fft2d
//!
//! This benchmark imports from bm3d_core - no code duplication.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array2;
use rand::prelude::*;
use rustfft::FftPlanner;

// Import from bm3d_core - the whole point of this restructure!
use bm3d_core::{
    Bm3dMode, Bm3dPlans,
    fft2d, ifft2d,
    wht2d_8x8_forward, wht2d_8x8_inverse,
    estimate_streak_profile_impl,
    run_bm3d_kernel,
};
use bm3d_core::block_matching::{compute_integral_images, find_similar_patches};

// =============================================================================
// Helper Functions for Test Data Generation
// =============================================================================

fn random_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_fn((rows, cols), |_| rng.gen())
}

// =============================================================================
// FFT Benchmarks
// =============================================================================

fn bench_fft2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft2d");

    for size in [8, 16, 32, 64, 128, 256] {
        let input = random_matrix(size, size, 42);
        let mut planner = FftPlanner::new();
        let fft_row = planner.plan_fft_forward(size);
        let fft_col = planner.plan_fft_forward(size);
        let ifft_row = planner.plan_fft_inverse(size);
        let ifft_col = planner.plan_fft_inverse(size);

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(
            BenchmarkId::new("forward", size),
            &size,
            |b, _| {
                b.iter(|| fft2d(black_box(input.view()), &fft_row, &fft_col))
            },
        );

        let freq = fft2d(input.view(), &fft_row, &fft_col);
        group.bench_with_input(
            BenchmarkId::new("inverse", size),
            &size,
            |b, _| {
                b.iter(|| ifft2d(black_box(&freq), &ifft_row, &ifft_col))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("roundtrip", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let f = fft2d(black_box(input.view()), &fft_row, &fft_col);
                    ifft2d(&f, &ifft_row, &ifft_col)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// WHT Benchmarks
// =============================================================================

fn bench_wht_8x8(c: &mut Criterion) {
    let mut group = c.benchmark_group("wht_8x8");
    let input = random_matrix(8, 8, 123);

    group.throughput(Throughput::Elements(64));

    group.bench_function("forward", |b| {
        b.iter(|| wht2d_8x8_forward(black_box(input.view())))
    });

    let freq = wht2d_8x8_forward(input.view());
    group.bench_function("inverse", |b| {
        b.iter(|| wht2d_8x8_inverse(black_box(&freq)))
    });

    group.bench_function("roundtrip", |b| {
        b.iter(|| {
            let f = wht2d_8x8_forward(black_box(input.view()));
            wht2d_8x8_inverse(&f)
        })
    });

    group.finish();
}

// =============================================================================
// Block Matching Benchmarks
// =============================================================================

fn bench_block_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_matching");

    for size in [64, 128, 256] {
        let image = random_matrix(size, size, 42);
        let (integral_sum, integral_sq_sum) = compute_integral_images(image.view());

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(
            BenchmarkId::new("integral_images", size),
            &size,
            |b, _| {
                b.iter(|| compute_integral_images(black_box(image.view())))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("find_similar_8x8_win24_max16", size),
            &size,
            |b, _| {
                b.iter(|| {
                    find_similar_patches(
                        black_box(image.view()),
                        &integral_sum,
                        &integral_sq_sum,
                        (size / 2, size / 2),
                        (8, 8),
                        (24, 24),
                        16,
                        2,
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("find_similar_8x8_win40_max64", size),
            &size,
            |b, _| {
                b.iter(|| {
                    find_similar_patches(
                        black_box(image.view()),
                        &integral_sum,
                        &integral_sq_sum,
                        (size / 2, size / 2),
                        (8, 8),
                        (40, 40),
                        64,
                        2,
                    )
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Streak Profile Benchmarks
// =============================================================================

fn bench_streak_profile(c: &mut Criterion) {
    let mut group = c.benchmark_group("streak_profile");
    group.sample_size(20);

    for (rows, cols) in [(180, 256), (360, 512)] {
        let sinogram = random_matrix(rows, cols, 42);

        group.throughput(Throughput::Elements((rows * cols) as u64));

        group.bench_with_input(
            BenchmarkId::new("iter1", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, _| {
                b.iter(|| {
                    estimate_streak_profile_impl(
                        black_box(sinogram.view()),
                        3.0,
                        1,
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("iter3", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, _| {
                b.iter(|| {
                    estimate_streak_profile_impl(
                        black_box(sinogram.view()),
                        3.0,
                        3,
                    )
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Full BM3D Pipeline Benchmarks
// =============================================================================

fn bench_bm3d_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("bm3d_full");
    group.sample_size(10);

    for (size, label) in [(128, "small_128"), (256, "medium_256")] {
        let image = random_matrix(size, size, 42);
        let sigma_psd = Array2::<f32>::zeros((8, 8));
        let sigma_map = Array2::<f32>::zeros((1, 1));
        let plans = Bm3dPlans::new(8, 64);

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(
            BenchmarkId::new("hard_threshold", label),
            &size,
            |b, _| {
                b.iter(|| {
                    run_bm3d_kernel(
                        black_box(image.view()),
                        image.view(),
                        Bm3dMode::HardThreshold,
                        sigma_psd.view(),
                        sigma_map.view(),
                        0.1,
                        2.7,
                        8,
                        2,
                        24,
                        64,
                        &plans,
                    )
                })
            },
        );

        let pilot = run_bm3d_kernel(
            image.view(),
            image.view(),
            Bm3dMode::HardThreshold,
            sigma_psd.view(),
            sigma_map.view(),
            0.1,
            2.7,
            8,
            2,
            24,
            64,
            &plans,
        );

        group.bench_with_input(
            BenchmarkId::new("wiener", label),
            &size,
            |b, _| {
                b.iter(|| {
                    run_bm3d_kernel(
                        black_box(image.view()),
                        pilot.view(),
                        Bm3dMode::Wiener,
                        sigma_psd.view(),
                        sigma_map.view(),
                        0.1,
                        0.0,
                        8,
                        2,
                        24,
                        64,
                        &plans,
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("full_2pass", label),
            &size,
            |b, _| {
                b.iter(|| {
                    let ht_result = run_bm3d_kernel(
                        black_box(image.view()),
                        image.view(),
                        Bm3dMode::HardThreshold,
                        sigma_psd.view(),
                        sigma_map.view(),
                        0.1,
                        2.7,
                        8,
                        2,
                        24,
                        64,
                        &plans,
                    );
                    run_bm3d_kernel(
                        image.view(),
                        ht_result.view(),
                        Bm3dMode::Wiener,
                        sigma_psd.view(),
                        sigma_map.view(),
                        0.1,
                        0.0,
                        8,
                        2,
                        24,
                        64,
                        &plans,
                    )
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    benches,
    bench_fft2d,
    bench_wht_8x8,
    bench_block_matching,
    bench_streak_profile,
    bench_bm3d_full,
);

criterion_main!(benches);
