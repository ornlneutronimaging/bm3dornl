use bm3d_core::{bm3d_ring_artifact_removal, Bm3dConfig, RingRemovalMode};
use ndarray::Array2;
use std::time::Instant;

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse::<T>().ok())
        .unwrap_or(default)
}

fn build_sinogram(rows: usize, cols: usize) -> Array2<f32> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    Array2::from_shape_fn((rows, cols), |_| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32)
    })
}

fn peak_rss_mb() -> Option<f64> {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if rc != 0 {
        return None;
    }
    #[cfg(target_os = "macos")]
    {
        Some(usage.ru_maxrss as f64 / (1024.0 * 1024.0))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Some(usage.ru_maxrss as f64 / 1024.0)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rows = parse_arg(&args, "--rows", 3273usize);
    let cols = parse_arg(&args, "--cols", 6200usize);
    let sigma_random = parse_arg(&args, "--sigma-random", 0.0f32);
    let patch_size = parse_arg(&args, "--patch-size", 8usize);
    let step_size = parse_arg(&args, "--step-size", 4usize);
    let search_window = parse_arg(&args, "--search-window", 24usize);
    let max_matches = parse_arg(&args, "--max-matches", 16usize);

    println!(
        "issue93 bench start rows={} cols={} sigma_random={} patch={} step={} search_window={} max_matches={} tile_env={:?}",
        rows,
        cols,
        sigma_random,
        patch_size,
        step_size,
        search_window,
        max_matches,
        std::env::var("BM3D_AGGREGATION_TILE_SIZE").ok()
    );

    let sinogram = build_sinogram(rows, cols);
    let config = Bm3dConfig::<f32> {
        sigma_random,
        patch_size,
        step_size,
        search_window,
        max_matches,
        ..Bm3dConfig::<f32>::default()
    };

    let rss_before_mb = peak_rss_mb().unwrap_or(0.0);

    let t0 = Instant::now();
    let out = bm3d_ring_artifact_removal(sinogram.view(), RingRemovalMode::Streak, &config)
        .expect("bm3d_ring_artifact_removal failed");
    let elapsed = t0.elapsed();
    let rss_peak_mb = peak_rss_mb().unwrap_or(0.0);

    // Keep output observable to avoid accidental optimization assumptions.
    let checksum: f64 = out
        .iter()
        .step_by((rows * cols / 4096).max(1))
        .map(|&v| v as f64)
        .sum();

    println!(
        "issue93 bench done elapsed_s={:.3} checksum={:.9} rss_before_mb={:.1} rss_peak_mb={:.1}",
        elapsed.as_secs_f64(),
        checksum,
        rss_before_mb,
        rss_peak_mb
    );
}
