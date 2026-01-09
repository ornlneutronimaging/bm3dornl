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
use bm3d_core::block_matching::{compute_integral_images, find_similar_patches};
use bm3d_core::{
    estimate_streak_profile_impl, fft2d, ifft2d, run_bm3d_kernel, wht2d_8x8_forward,
    wht2d_8x8_inverse, Bm3dMode, Bm3dPlans,
};

// =============================================================================
// Helper Functions for Test Data Generation
// =============================================================================

fn random_matrix_f32(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_fn((rows, cols), |_| rng.gen())
}

fn random_matrix_f64(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_fn((rows, cols), |_| rng.gen())
}

// =============================================================================
// FFT Benchmarks
// =============================================================================

fn bench_fft2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft2d");

    for size in [8, 16, 32, 64, 128, 256] {
        let input = random_matrix_f32(size, size, 42);
        let mut planner = FftPlanner::new();
        let fft_row = planner.plan_fft_forward(size);
        let fft_col = planner.plan_fft_forward(size);
        let ifft_row = planner.plan_fft_inverse(size);
        let ifft_col = planner.plan_fft_inverse(size);

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::new("forward", size), &size, |b, _| {
            b.iter(|| fft2d(black_box(input.view()), &fft_row, &fft_col))
        });

        let freq = fft2d(input.view(), &fft_row, &fft_col);
        group.bench_with_input(BenchmarkId::new("inverse", size), &size, |b, _| {
            b.iter(|| ifft2d(black_box(&freq), &ifft_row, &ifft_col))
        });

        group.bench_with_input(BenchmarkId::new("roundtrip", size), &size, |b, _| {
            b.iter(|| {
                let f = fft2d(black_box(input.view()), &fft_row, &fft_col);
                ifft2d(&f, &ifft_row, &ifft_col)
            })
        });
    }

    group.finish();
}

// =============================================================================
// WHT Benchmarks
// =============================================================================

fn bench_wht_8x8(c: &mut Criterion) {
    let mut group = c.benchmark_group("wht_8x8");
    let input = random_matrix_f32(8, 8, 123);

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
        let image = random_matrix_f32(size, size, 42);
        let (integral_sum, integral_sq_sum) = compute_integral_images(image.view());

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::new("integral_images", size), &size, |b, _| {
            b.iter(|| compute_integral_images(black_box(image.view())))
        });

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
        let sinogram_f32 = random_matrix_f32(rows, cols, 42);
        let sinogram_f64 = random_matrix_f64(rows, cols, 42);

        group.throughput(Throughput::Elements((rows * cols) as u64));

        // f32 benchmarks
        group.bench_with_input(
            BenchmarkId::new("f32/iter1", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, _| {
                b.iter(|| estimate_streak_profile_impl(black_box(sinogram_f32.view()), 3.0f32, 1))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("f32/iter3", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, _| {
                b.iter(|| estimate_streak_profile_impl(black_box(sinogram_f32.view()), 3.0f32, 3))
            },
        );

        // f64 benchmarks
        group.bench_with_input(
            BenchmarkId::new("f64/iter1", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, _| {
                b.iter(|| estimate_streak_profile_impl(black_box(sinogram_f64.view()), 3.0f64, 1))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("f64/iter3", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, _| {
                b.iter(|| estimate_streak_profile_impl(black_box(sinogram_f64.view()), 3.0f64, 3))
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
        let image = random_matrix_f32(size, size, 42);
        let sigma_psd = Array2::<f32>::zeros((8, 8));
        let sigma_map = Array2::<f32>::zeros((1, 1));
        let plans: Bm3dPlans<f32> = Bm3dPlans::new(8, 64);

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::new("hard_threshold", label), &size, |b, _| {
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
        });

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

        group.bench_with_input(BenchmarkId::new("wiener", label), &size, |b, _| {
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
        });

        group.bench_with_input(BenchmarkId::new("full_2pass", label), &size, |b, _| {
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
        });
    }

    group.finish();
}

// =============================================================================
// f32 vs f64 Precision Comparison Benchmarks
// =============================================================================

fn bench_precision_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32_vs_f64");
    group.sample_size(10);

    // Benchmark full BM3D pipeline for both precisions
    let size = 128;
    let image_f32 = random_matrix_f32(size, size, 42);
    let image_f64 = random_matrix_f64(size, size, 42);
    let sigma_psd_f32 = Array2::<f32>::zeros((8, 8));
    let sigma_psd_f64 = Array2::<f64>::zeros((8, 8));
    let sigma_map_f32 = Array2::<f32>::zeros((1, 1));
    let sigma_map_f64 = Array2::<f64>::zeros((1, 1));
    let plans_f32: Bm3dPlans<f32> = Bm3dPlans::new(8, 64);
    let plans_f64: Bm3dPlans<f64> = Bm3dPlans::new(8, 64);

    group.throughput(Throughput::Elements((size * size) as u64));

    // f32 full 2-pass pipeline
    group.bench_function("bm3d_128_f32", |b| {
        b.iter(|| {
            let ht_result = run_bm3d_kernel(
                black_box(image_f32.view()),
                image_f32.view(),
                Bm3dMode::HardThreshold,
                sigma_psd_f32.view(),
                sigma_map_f32.view(),
                0.1f32,
                2.7f32,
                8,
                2,
                24,
                64,
                &plans_f32,
            );
            run_bm3d_kernel(
                image_f32.view(),
                ht_result.view(),
                Bm3dMode::Wiener,
                sigma_psd_f32.view(),
                sigma_map_f32.view(),
                0.1f32,
                0.0f32,
                8,
                2,
                24,
                64,
                &plans_f32,
            )
        })
    });

    // f64 full 2-pass pipeline
    group.bench_function("bm3d_128_f64", |b| {
        b.iter(|| {
            let ht_result = run_bm3d_kernel(
                black_box(image_f64.view()),
                image_f64.view(),
                Bm3dMode::HardThreshold,
                sigma_psd_f64.view(),
                sigma_map_f64.view(),
                0.1f64,
                2.7f64,
                8,
                2,
                24,
                64,
                &plans_f64,
            );
            run_bm3d_kernel(
                image_f64.view(),
                ht_result.view(),
                Bm3dMode::Wiener,
                sigma_psd_f64.view(),
                sigma_map_f64.view(),
                0.1f64,
                0.0f64,
                8,
                2,
                24,
                64,
                &plans_f64,
            )
        })
    });

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
    bench_precision_comparison,
);

criterion_main!(benches);
