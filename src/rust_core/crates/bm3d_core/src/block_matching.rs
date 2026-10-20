use ndarray::{Array2, ArrayView2};
use std::any::TypeId;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::cmp::Ordering;

use crate::float_trait::Bm3dFloat;

#[derive(Debug, Clone, Copy)]
pub struct PatchMatch<F: Bm3dFloat> {
    pub row: usize,
    pub col: usize,
    pub distance: F,
}

impl<F: Bm3dFloat> PartialEq for PatchMatch<F> {
    fn eq(&self, other: &Self) -> bool {
        self.row == other.row && self.col == other.col && self.distance == other.distance
    }
}

impl<F: Bm3dFloat> Eq for PatchMatch<F> {}

impl<F: Bm3dFloat> Ord for PatchMatch<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl<F: Bm3dFloat> PartialOrd for PatchMatch<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[inline(always)]
fn compute_squared_distance_at<F: Bm3dFloat>(
    image: ArrayView2<F>,
    ref_r: usize,
    ref_c: usize,
    cand_r: usize,
    cand_c: usize,
    ph: usize,
    pw: usize,
    threshold: F,
) -> F {
    let mut sum_sq = F::zero();
    for dr in 0..ph {
        let ref_row = image.row(ref_r + dr);
        let cand_row = image.row(cand_r + dr);
        for dc in 0..pw {
            let diff = ref_row[ref_c + dc] - cand_row[cand_c + dc];
            sum_sq += diff * diff;
        }
        if sum_sq >= threshold {
            return sum_sq;
        }
    }
    sum_sq
}

#[inline(always)]
fn compute_squared_distance_at_strided<F: Bm3dFloat>(
    image_data: &[F],
    image_cols: usize,
    ref_r: usize,
    ref_c: usize,
    cand_r: usize,
    cand_c: usize,
    ph: usize,
    pw: usize,
    threshold: F,
) -> F {
    let mut sum_sq = F::zero();
    for dr in 0..ph {
        let ref_base = (ref_r + dr) * image_cols + ref_c;
        let cand_base = (cand_r + dr) * image_cols + cand_c;
        for dc in 0..pw {
            let diff = image_data[ref_base + dc] - image_data[cand_base + dc];
            sum_sq += diff * diff;
        }
        if sum_sq >= threshold {
            return sum_sq;
        }
    }
    sum_sq
}

#[inline(always)]
fn compute_squared_distance_at_strided_8x8_scalar<F: Bm3dFloat>(
    image_data: &[F],
    image_cols: usize,
    ref_r: usize,
    ref_c: usize,
    cand_r: usize,
    cand_c: usize,
    threshold: F,
) -> F {
    let mut sum_sq = F::zero();
    for dr in 0..8 {
        let ref_base = (ref_r + dr) * image_cols + ref_c;
        let cand_base = (cand_r + dr) * image_cols + cand_c;

        let d0 = image_data[ref_base] - image_data[cand_base];
        let d1 = image_data[ref_base + 1] - image_data[cand_base + 1];
        let d2 = image_data[ref_base + 2] - image_data[cand_base + 2];
        let d3 = image_data[ref_base + 3] - image_data[cand_base + 3];
        let d4 = image_data[ref_base + 4] - image_data[cand_base + 4];
        let d5 = image_data[ref_base + 5] - image_data[cand_base + 5];
        let d6 = image_data[ref_base + 6] - image_data[cand_base + 6];
        let d7 = image_data[ref_base + 7] - image_data[cand_base + 7];

        sum_sq += d0 * d0;
        sum_sq += d1 * d1;
        sum_sq += d2 * d2;
        sum_sq += d3 * d3;
        sum_sq += d4 * d4;
        sum_sq += d5 * d5;
        sum_sq += d6 * d6;
        sum_sq += d7 * d7;

        if sum_sq >= threshold {
            return sum_sq;
        }
    }
    sum_sq
}

#[inline(always)]
fn compute_squared_distance_at_strided_8x8_f32_dispatch(
    image_data: &[f32],
    image_cols: usize,
    ref_r: usize,
    ref_c: usize,
    cand_r: usize,
    cand_c: usize,
    threshold: f32,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: Guarded by runtime feature check.
            unsafe {
                compute_squared_distance_at_strided_8x8_f32_avx2(
                    image_data, image_cols, ref_r, ref_c, cand_r, cand_c, threshold,
                )
            }
        } else {
            // SAFETY: SSE2 is guaranteed on x86_64.
            unsafe {
                compute_squared_distance_at_strided_8x8_f32_sse2(
                    image_data, image_cols, ref_r, ref_c, cand_r, cand_c, threshold,
                )
            }
        }
    }

    #[cfg(target_arch = "x86")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: Guarded by runtime feature check.
            unsafe {
                compute_squared_distance_at_strided_8x8_f32_avx2(
                    image_data, image_cols, ref_r, ref_c, cand_r, cand_c, threshold,
                )
            }
        } else if std::arch::is_x86_feature_detected!("sse2") {
            // SAFETY: Guarded by runtime feature check.
            unsafe {
                compute_squared_distance_at_strided_8x8_f32_sse2(
                    image_data, image_cols, ref_r, ref_c, cand_r, cand_c, threshold,
                )
            }
        } else {
            compute_squared_distance_at_strided_8x8_scalar(
                image_data, image_cols, ref_r, ref_c, cand_r, cand_c, threshold,
            )
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe {
            compute_squared_distance_at_strided_8x8_f32_neon(
                image_data, image_cols, ref_r, ref_c, cand_r, cand_c, threshold,
            )
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
    {
        compute_squared_distance_at_strided_8x8_scalar(
            image_data, image_cols, ref_r, ref_c, cand_r, cand_c, threshold,
        )
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn compute_squared_distance_at_strided_8x8_f32_sse2(
    image_data: &[f32],
    image_cols: usize,
    ref_r: usize,
    ref_c: usize,
    cand_r: usize,
    cand_c: usize,
    threshold: f32,
) -> f32 {
    // SAFETY: SSE2 is guaranteed baseline on x86_64 and runtime-checked on x86 by
    // `compute_squared_distance_at_strided_8x8_f32_dispatch` before this function is
    // called.
    unsafe {
        let mut sum_sq = 0.0f32;
        for dr in 0..8 {
            let ref_base = (ref_r + dr) * image_cols + ref_c;
            let cand_base = (cand_r + dr) * image_cols + cand_c;
            let ref_ptr = image_data.as_ptr().add(ref_base);
            let cand_ptr = image_data.as_ptr().add(cand_base);

            let ref_lo = _mm_loadu_ps(ref_ptr);
            let cand_lo = _mm_loadu_ps(cand_ptr);
            let diff_lo = _mm_sub_ps(ref_lo, cand_lo);
            let sq_lo = _mm_mul_ps(diff_lo, diff_lo);

            let ref_hi = _mm_loadu_ps(ref_ptr.add(4));
            let cand_hi = _mm_loadu_ps(cand_ptr.add(4));
            let diff_hi = _mm_sub_ps(ref_hi, cand_hi);
            let sq_hi = _mm_mul_ps(diff_hi, diff_hi);

            let mut lanes_lo = [0.0f32; 4];
            let mut lanes_hi = [0.0f32; 4];
            _mm_storeu_ps(lanes_lo.as_mut_ptr(), sq_lo);
            _mm_storeu_ps(lanes_hi.as_mut_ptr(), sq_hi);

            sum_sq += lanes_lo[0]
                + lanes_lo[1]
                + lanes_lo[2]
                + lanes_lo[3]
                + lanes_hi[0]
                + lanes_hi[1]
                + lanes_hi[2]
                + lanes_hi[3];
            if sum_sq >= threshold {
                return sum_sq;
            }
        }
        sum_sq
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn compute_squared_distance_at_strided_8x8_f32_avx2(
    image_data: &[f32],
    image_cols: usize,
    ref_r: usize,
    ref_c: usize,
    cand_r: usize,
    cand_c: usize,
    threshold: f32,
) -> f32 {
    // SAFETY: The required SIMD feature (AVX2) is runtime-checked by
    // `compute_squared_distance_at_strided_8x8_f32_dispatch` before this function is
    // called.
    unsafe {
        let mut sum_sq = 0.0f32;
        for dr in 0..8 {
            let ref_base = (ref_r + dr) * image_cols + ref_c;
            let cand_base = (cand_r + dr) * image_cols + cand_c;
            let ref_ptr = image_data.as_ptr().add(ref_base);
            let cand_ptr = image_data.as_ptr().add(cand_base);

            let ref_vec = _mm256_loadu_ps(ref_ptr);
            let cand_vec = _mm256_loadu_ps(cand_ptr);
            let diff = _mm256_sub_ps(ref_vec, cand_vec);
            let sq = _mm256_mul_ps(diff, diff);

            let mut lanes = [0.0f32; 8];
            _mm256_storeu_ps(lanes.as_mut_ptr(), sq);
            sum_sq += lanes[0]
                + lanes[1]
                + lanes[2]
                + lanes[3]
                + lanes[4]
                + lanes[5]
                + lanes[6]
                + lanes[7];
            if sum_sq >= threshold {
                return sum_sq;
            }
        }
        sum_sq
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn compute_squared_distance_at_strided_8x8_f32_neon(
    image_data: &[f32],
    image_cols: usize,
    ref_r: usize,
    ref_c: usize,
    cand_r: usize,
    cand_c: usize,
    threshold: f32,
) -> f32 {
    unsafe {
        let mut sum_sq = 0.0f32;
        for dr in 0..8 {
            let ref_base = (ref_r + dr) * image_cols + ref_c;
            let cand_base = (cand_r + dr) * image_cols + cand_c;
            let ref_ptr = image_data.as_ptr().add(ref_base);
            let cand_ptr = image_data.as_ptr().add(cand_base);

            let ref_lo = vld1q_f32(ref_ptr);
            let cand_lo = vld1q_f32(cand_ptr);
            let diff_lo = vsubq_f32(ref_lo, cand_lo);
            let sq_lo = vmulq_f32(diff_lo, diff_lo);

            let ref_hi = vld1q_f32(ref_ptr.add(4));
            let cand_hi = vld1q_f32(cand_ptr.add(4));
            let diff_hi = vsubq_f32(ref_hi, cand_hi);
            let sq_hi = vmulq_f32(diff_hi, diff_hi);

            sum_sq += vaddvq_f32(sq_lo) + vaddvq_f32(sq_hi);
            if sum_sq >= threshold {
                return sum_sq;
            }
        }
        sum_sq
    }
}

#[inline(always)]
fn compute_squared_distance_at_strided_8x8<F: Bm3dFloat>(
    image_data: &[F],
    image_cols: usize,
    ref_r: usize,
    ref_c: usize,
    cand_r: usize,
    cand_c: usize,
    threshold: F,
) -> F {
    if TypeId::of::<F>() == TypeId::of::<f32>() {
        // SAFETY: TypeId check guarantees F is f32 in this branch.
        let data_f32 = unsafe {
            std::slice::from_raw_parts(image_data.as_ptr() as *const f32, image_data.len())
        };
        let threshold_f32 = threshold.to_f32().unwrap_or(f32::MAX);
        let sum = compute_squared_distance_at_strided_8x8_f32_dispatch(
            data_f32,
            image_cols,
            ref_r,
            ref_c,
            cand_r,
            cand_c,
            threshold_f32,
        );
        return F::from_f64_c(sum as f64);
    }
    compute_squared_distance_at_strided_8x8_scalar(
        image_data, image_cols, ref_r, ref_c, cand_r, cand_c, threshold,
    )
}

/// Returns an array with shape `(h + 1, w + 1)`.
pub fn compute_integral_sum_image<F: Bm3dFloat>(image: ArrayView2<F>) -> Array2<F> {
    let (h, w) = image.dim();
    let mut sum_img = Array2::<F>::zeros((h + 1, w + 1));

    for r in 0..h {
        let mut row_sum = F::zero();
        for c in 0..w {
            let val = image[[r, c]];
            row_sum += val;
            sum_img[[r + 1, c + 1]] = sum_img[[r, c + 1]] + row_sum;
        }
    }
    sum_img
}

pub fn compute_integral_images<F: Bm3dFloat>(image: ArrayView2<F>) -> (Array2<F>, Array2<F>) {
    let (h, w) = image.dim();
    let sum_img = compute_integral_sum_image(image);
    let mut sq_sum_img = Array2::<F>::zeros((h + 1, w + 1));

    for r in 0..h {
        let mut row_sq_sum = F::zero();
        for c in 0..w {
            let val = image[[r, c]];
            row_sq_sum += val * val;

            sq_sum_img[[r + 1, c + 1]] = sq_sum_img[[r, c + 1]] + row_sq_sum;
        }
    }
    (sum_img, sq_sum_img)
}

#[inline(always)]
fn get_patch_sum<F: Bm3dFloat>(sum_img: &Array2<F>, r: usize, c: usize, h: usize, w: usize) -> F {
    let r1 = r;
    let c1 = c;
    let r2 = r + h;
    let c2 = c + w;

    sum_img[[r2, c2]] - sum_img[[r1, c2]] - sum_img[[r2, c1]] + sum_img[[r1, c1]]
}

#[inline(always)]
fn get_patch_sum_strided<F: Bm3dFloat>(
    sum_data: &[F],
    sum_stride: usize,
    r: usize,
    c: usize,
    h: usize,
    w: usize,
) -> F {
    let r1 = r;
    let c1 = c;
    let r2 = r + h;
    let c2 = c + w;
    sum_data[r2 * sum_stride + c2] - sum_data[r1 * sum_stride + c2] - sum_data[r2 * sum_stride + c1]
        + sum_data[r1 * sum_stride + c1]
}

/// Find similar patches within a search window using Integral Image pre-screening (sum bound).
pub fn find_similar_patches_in_place_sum<F: Bm3dFloat>(
    image: ArrayView2<F>,
    integral_sum: &Array2<F>,
    ref_pos: (usize, usize),
    patch_size: (usize, usize),
    search_window: (usize, usize),
    max_matches: usize,
    step: usize,
    out_matches: &mut Vec<PatchMatch<F>>,
) {
    let (ref_r, ref_c) = ref_pos;
    let (ph, pw) = patch_size;
    let (h, w) = image.dim();

    let search_r_start = ref_r.saturating_sub(search_window.0 / 2);
    let search_r_end = (ref_r + search_window.0 / 2).min(h - ph);
    let search_c_start = ref_c.saturating_sub(search_window.1 / 2);
    let search_c_end = (ref_c + search_window.1 / 2).min(w - pw);

    out_matches.clear();
    out_matches.push(PatchMatch {
        row: ref_r,
        col: ref_c,
        distance: F::zero(),
    });

    let mut threshold = if max_matches > 1 {
        F::max_value()
    } else {
        F::zero()
    };

    let inv_n = F::one() / F::usize_as(ph * pw);
    // The fast path indexes both arrays as row-major slices, so it needs
    // standard layout (`as_slice`); any other layout takes the fallback.
    if let (Some(image_data), Some(sum_data)) = (image.as_slice(), integral_sum.as_slice()) {
        let sum_stride = integral_sum.dim().1;
        let image_cols = w;
        let ref_sum = get_patch_sum_strided(sum_data, sum_stride, ref_r, ref_c, ph, pw);
        for r in (search_r_start..=search_r_end).step_by(step) {
            for c in (search_c_start..=search_c_end).step_by(step) {
                if r == ref_r && c == ref_c {
                    continue;
                }

                let cand_sum = get_patch_sum_strided(sum_data, sum_stride, r, c, ph, pw);
                let check_threshold = threshold;

                let diff_sum = cand_sum - ref_sum;
                let lb_mean = (diff_sum * diff_sum) * inv_n;
                if lb_mean >= check_threshold {
                    continue;
                }

                let dist = if ph == 8 && pw == 8 {
                    compute_squared_distance_at_strided_8x8(
                        image_data, image_cols, ref_r, ref_c, r, c, threshold,
                    )
                } else {
                    compute_squared_distance_at_strided(
                        image_data, image_cols, ref_r, ref_c, r, c, ph, pw, threshold,
                    )
                };

                if dist < threshold {
                    if out_matches.len() < max_matches {
                        out_matches.push(PatchMatch {
                            row: r,
                            col: c,
                            distance: dist,
                        });
                        if out_matches.len() == max_matches {
                            let mut worst_idx = 0usize;
                            let mut worst_dist = out_matches[0].distance;
                            for (idx, m) in out_matches.iter().enumerate().skip(1) {
                                if m.distance > worst_dist {
                                    worst_idx = idx;
                                    worst_dist = m.distance;
                                }
                            }
                            threshold = out_matches[worst_idx].distance;
                        }
                    } else {
                        let mut worst_idx = 0usize;
                        let mut worst_dist = out_matches[0].distance;
                        for (idx, m) in out_matches.iter().enumerate().skip(1) {
                            if m.distance > worst_dist {
                                worst_idx = idx;
                                worst_dist = m.distance;
                            }
                        }
                        out_matches[worst_idx] = PatchMatch {
                            row: r,
                            col: c,
                            distance: dist,
                        };
                        let mut next_worst = out_matches[0].distance;
                        for m in out_matches.iter().skip(1) {
                            if m.distance > next_worst {
                                next_worst = m.distance;
                            }
                        }
                        threshold = next_worst;
                    }
                }
            }
        }
    } else {
        let ref_sum = get_patch_sum(integral_sum, ref_r, ref_c, ph, pw);
        for r in (search_r_start..=search_r_end).step_by(step) {
            for c in (search_c_start..=search_c_end).step_by(step) {
                if r == ref_r && c == ref_c {
                    continue;
                }

                let cand_sum = get_patch_sum(integral_sum, r, c, ph, pw);
                let check_threshold = threshold;

                let diff_sum = cand_sum - ref_sum;
                let lb_mean = (diff_sum * diff_sum) * inv_n;
                if lb_mean >= check_threshold {
                    continue;
                }

                let dist =
                    compute_squared_distance_at(image, ref_r, ref_c, r, c, ph, pw, threshold);

                if dist < threshold {
                    if out_matches.len() < max_matches {
                        out_matches.push(PatchMatch {
                            row: r,
                            col: c,
                            distance: dist,
                        });
                        if out_matches.len() == max_matches {
                            let mut worst_idx = 0usize;
                            let mut worst_dist = out_matches[0].distance;
                            for (idx, m) in out_matches.iter().enumerate().skip(1) {
                                if m.distance > worst_dist {
                                    worst_idx = idx;
                                    worst_dist = m.distance;
                                }
                            }
                            threshold = out_matches[worst_idx].distance;
                        }
                    } else {
                        let mut worst_idx = 0usize;
                        let mut worst_dist = out_matches[0].distance;
                        for (idx, m) in out_matches.iter().enumerate().skip(1) {
                            if m.distance > worst_dist {
                                worst_idx = idx;
                                worst_dist = m.distance;
                            }
                        }
                        out_matches[worst_idx] = PatchMatch {
                            row: r,
                            col: c,
                            distance: dist,
                        };
                        let mut next_worst = out_matches[0].distance;
                        for m in out_matches.iter().skip(1) {
                            if m.distance > next_worst {
                                next_worst = m.distance;
                            }
                        }
                        threshold = next_worst;
                    }
                }
            }
        }
    }

    out_matches.sort_unstable_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(Ordering::Equal)
    });
}

pub fn find_similar_patches_in_place<F: Bm3dFloat>(
    image: ArrayView2<F>,
    integral_sum: &Array2<F>,
    _integral_sq_sum: &Array2<F>,
    ref_pos: (usize, usize),
    patch_size: (usize, usize),
    search_window: (usize, usize),
    max_matches: usize,
    step: usize,
    out_matches: &mut Vec<PatchMatch<F>>,
) {
    find_similar_patches_in_place_sum(
        image,
        integral_sum,
        ref_pos,
        patch_size,
        search_window,
        max_matches,
        step,
        out_matches,
    );
}

/// Find similar patches within a search window using Integral Image Pre-Screening.
pub fn find_similar_patches<F: Bm3dFloat>(
    image: ArrayView2<F>,
    integral_sum: &Array2<F>,
    integral_sq_sum: &Array2<F>,
    ref_pos: (usize, usize),
    patch_size: (usize, usize),
    search_window: (usize, usize),
    max_matches: usize,
    step: usize,
) -> Vec<PatchMatch<F>> {
    let mut matches = Vec::with_capacity(max_matches.max(1));
    find_similar_patches_in_place(
        image,
        integral_sum,
        integral_sq_sum,
        ref_pos,
        patch_size,
        search_window,
        max_matches,
        step,
        &mut matches,
    );
    matches
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use std::collections::BinaryHeap;

    struct SimpleLcg {
        state: u64,
    }

    impl SimpleLcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.state
        }

        fn next_f32(&mut self) -> f32 {
            let u = self.next_u64();
            (u >> 40) as f32 / (1u64 << 24) as f32
        }

        fn next_f64(&mut self) -> f64 {
            let u = self.next_u64();
            (u >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn random_matrix_f32(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
        let mut rng = SimpleLcg::new(seed);
        Array2::from_shape_fn((rows, cols), |_| rng.next_f32())
    }

    fn random_matrix_f64(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
        let mut rng = SimpleLcg::new(seed);
        Array2::from_shape_fn((rows, cols), |_| rng.next_f64())
    }

    #[test]
    fn test_patch_match_ordering_by_distance() {
        let p1: PatchMatch<f32> = PatchMatch {
            row: 0,
            col: 0,
            distance: 1.0,
        };
        let p2: PatchMatch<f32> = PatchMatch {
            row: 1,
            col: 1,
            distance: 2.0,
        };
        let p3: PatchMatch<f32> = PatchMatch {
            row: 2,
            col: 2,
            distance: 0.5,
        };

        assert!(p3 < p1, "0.5 should be less than 1.0");
        assert!(p1 < p2, "1.0 should be less than 2.0");
        assert!(p3 < p2, "0.5 should be less than 2.0");
    }

    #[test]
    fn test_patch_match_heap_behavior() {
        let mut heap: BinaryHeap<PatchMatch<f32>> = BinaryHeap::new();
        heap.push(PatchMatch {
            row: 0,
            col: 0,
            distance: 1.0,
        });
        heap.push(PatchMatch {
            row: 1,
            col: 1,
            distance: 3.0,
        });
        heap.push(PatchMatch {
            row: 2,
            col: 2,
            distance: 2.0,
        });
        heap.push(PatchMatch {
            row: 3,
            col: 3,
            distance: 0.5,
        });

        let p1 = heap.pop().unwrap();
        assert_eq!(p1.distance, 3.0, "First pop should be largest distance");

        let p2 = heap.pop().unwrap();
        assert_eq!(p2.distance, 2.0, "Second pop should be second largest");

        let p3 = heap.pop().unwrap();
        assert_eq!(p3.distance, 1.0, "Third pop should be 1.0");

        let p4 = heap.pop().unwrap();
        assert_eq!(p4.distance, 0.5, "Fourth pop should be smallest");
    }

    #[test]
    fn test_patch_match_equal_distance() {
        let p1: PatchMatch<f32> = PatchMatch {
            row: 0,
            col: 0,
            distance: 1.0,
        };
        let p2: PatchMatch<f32> = PatchMatch {
            row: 5,
            col: 5,
            distance: 1.0,
        };

        assert_eq!(
            p1.cmp(&p2),
            Ordering::Equal,
            "Equal distances should compare as Equal"
        );
        assert_eq!(p1.partial_cmp(&p2), Some(Ordering::Equal));
    }

    #[test]
    fn test_patch_match_heap_with_equal_distances() {
        let mut heap: BinaryHeap<PatchMatch<f32>> = BinaryHeap::new();
        heap.push(PatchMatch {
            row: 0,
            col: 0,
            distance: 1.0,
        });
        heap.push(PatchMatch {
            row: 1,
            col: 1,
            distance: 1.0,
        });
        heap.push(PatchMatch {
            row: 2,
            col: 2,
            distance: 1.0,
        });

        assert_eq!(heap.len(), 3);

        for _ in 0..3 {
            let p = heap.pop().unwrap();
            assert_eq!(p.distance, 1.0);
        }
    }

    #[test]
    fn test_integral_image_simple() {
        let mut input = Array2::<f32>::zeros((2, 2));
        input[[0, 0]] = 1.0;
        input[[0, 1]] = 2.0;
        input[[1, 0]] = 3.0;
        input[[1, 1]] = 4.0;

        let (sum_img, sq_sum_img) = compute_integral_images(input.view());

        assert_eq!(sum_img.dim(), (3, 3));
        assert_eq!(sum_img[[0, 0]], 0.0);
        assert_eq!(sum_img[[0, 1]], 0.0);
        assert_eq!(sum_img[[0, 2]], 0.0);
        assert_eq!(sum_img[[1, 0]], 0.0);
        assert_eq!(sum_img[[1, 1]], 1.0);
        assert_eq!(sum_img[[1, 2]], 3.0);
        assert_eq!(sum_img[[2, 0]], 0.0);
        assert_eq!(sum_img[[2, 1]], 4.0);
        assert_eq!(sum_img[[2, 2]], 10.0);

        assert_eq!(sq_sum_img[[1, 1]], 1.0);
        assert_eq!(sq_sum_img[[1, 2]], 5.0);
        assert_eq!(sq_sum_img[[2, 1]], 10.0);
        assert_eq!(sq_sum_img[[2, 2]], 30.0);
    }

    #[test]
    fn test_integral_image_simple_f64() {
        let mut input = Array2::<f64>::zeros((2, 2));
        input[[0, 0]] = 1.0;
        input[[0, 1]] = 2.0;
        input[[1, 0]] = 3.0;
        input[[1, 1]] = 4.0;

        let (sum_img, sq_sum_img) = compute_integral_images(input.view());

        assert_eq!(sum_img[[2, 2]], 10.0);
        assert_eq!(sq_sum_img[[2, 2]], 30.0);
    }

    #[test]
    fn test_integral_image_zeros() {
        let input = Array2::<f32>::zeros((4, 4));
        let (sum_img, sq_sum_img) = compute_integral_images(input.view());

        for val in sum_img.iter() {
            assert_eq!(*val, 0.0, "Integral of zeros should be all zeros");
        }
        for val in sq_sum_img.iter() {
            assert_eq!(*val, 0.0, "Squared integral of zeros should be all zeros");
        }
    }

    #[test]
    fn test_integral_image_ones() {
        let input = Array2::<f32>::ones((4, 4));
        let (sum_img, _) = compute_integral_images(input.view());

        for r in 0..4 {
            for c in 0..4 {
                let expected = ((r + 1) * (c + 1)) as f32;
                assert_eq!(
                    sum_img[[r + 1, c + 1]],
                    expected,
                    "Integral of ones at [{},{}] should be {}",
                    r + 1,
                    c + 1,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_integral_image_single_element() {
        let mut input = Array2::<f32>::zeros((1, 1));
        input[[0, 0]] = 5.0;

        let (sum_img, sq_sum_img) = compute_integral_images(input.view());

        assert_eq!(sum_img.dim(), (2, 2));
        assert_eq!(sum_img[[0, 0]], 0.0);
        assert_eq!(sum_img[[0, 1]], 0.0);
        assert_eq!(sum_img[[1, 0]], 0.0);
        assert_eq!(sum_img[[1, 1]], 5.0);

        assert_eq!(sq_sum_img[[1, 1]], 25.0);
    }

    #[test]
    fn test_integral_image_rectangular() {
        let mut input = Array2::<f32>::zeros((2, 3));
        for r in 0..2 {
            for c in 0..3 {
                input[[r, c]] = (r * 3 + c + 1) as f32;
            }
        }

        let (sum_img, _) = compute_integral_images(input.view());

        assert_eq!(sum_img.dim(), (3, 4));
        assert_eq!(sum_img[[2, 3]], 21.0);
        assert_eq!(sum_img[[1, 3]], 6.0);
        assert_eq!(sum_img[[2, 1]], 5.0);
    }

    #[test]
    fn test_integral_image_large_values() {
        let mut input = Array2::<f32>::zeros((4, 4));
        for r in 0..4 {
            for c in 0..4 {
                input[[r, c]] = 1e6;
            }
        }

        let (sum_img, sq_sum_img) = compute_integral_images(input.view());

        let expected_sum = 16.0 * 1e6;
        assert!(
            (sum_img[[4, 4]] - expected_sum).abs() < 1.0,
            "Large value sum should be correct: got {}, expected {}",
            sum_img[[4, 4]],
            expected_sum
        );

        let expected_sq_sum = 16.0 * 1e12;
        let rel_err = (sq_sum_img[[4, 4]] - expected_sq_sum).abs() / expected_sq_sum;
        assert!(
            rel_err < 1e-5,
            "Large value squared sum should be correct: got {}, expected {}, rel_err={}",
            sq_sum_img[[4, 4]],
            expected_sq_sum,
            rel_err
        );
    }

    #[test]
    fn test_find_similar_identical_image() {
        let image = Array2::<f32>::from_elem((16, 16), 0.5);
        let (sum_img, sq_sum_img) = compute_integral_images(image.view());

        let matches = find_similar_patches(
            image.view(),
            &sum_img,
            &sq_sum_img,
            (4, 4),
            (4, 4),
            (8, 8),
            5,
            1,
        );

        for m in &matches {
            assert!(
                m.distance < 1e-5,
                "Uniform image should have all distances ≈ 0, got {}",
                m.distance
            );
        }
    }

    #[test]
    fn test_find_similar_self_match() {
        let image = random_matrix_f32(16, 16, 12345);
        let (sum_img, sq_sum_img) = compute_integral_images(image.view());

        let matches = find_similar_patches(
            image.view(),
            &sum_img,
            &sq_sum_img,
            (4, 4),
            (4, 4),
            (8, 8),
            10,
            1,
        );

        let self_match = matches.iter().find(|m| m.row == 4 && m.col == 4);
        assert!(self_match.is_some(), "Self-match should be in results");
        assert_eq!(
            self_match.unwrap().distance,
            0.0,
            "Self-match distance should be 0"
        );

        assert_eq!(matches[0].row, 4);
        assert_eq!(matches[0].col, 4);
        assert_eq!(matches[0].distance, 0.0);
    }

    #[test]
    fn test_find_similar_f64() {
        let image = random_matrix_f64(16, 16, 12345);
        let (sum_img, sq_sum_img) = compute_integral_images(image.view());

        let matches = find_similar_patches(
            image.view(),
            &sum_img,
            &sq_sum_img,
            (4, 4),
            (4, 4),
            (8, 8),
            10,
            1,
        );

        assert_eq!(matches[0].row, 4);
        assert_eq!(matches[0].col, 4);
        assert_eq!(matches[0].distance, 0.0);
    }

    #[test]
    fn test_find_similar_distinct_regions() {
        let mut image = Array2::<f32>::zeros((16, 16));
        for r in 0..16 {
            for c in 8..16 {
                image[[r, c]] = 1.0;
            }
        }
        let (sum_img, sq_sum_img) = compute_integral_images(image.view());

        let matches = find_similar_patches(
            image.view(),
            &sum_img,
            &sq_sum_img,
            (4, 2),
            (4, 4),
            (16, 16),
            5,
            1,
        );

        for m in &matches {
            assert!(
                m.col <= 4,
                "Best matches for left region should be in left region, got col={}",
                m.col
            );
        }
    }

    #[test]
    fn test_find_similar_respects_max_matches() {
        let image = random_matrix_f32(32, 32, 54321);
        let (sum_img, sq_sum_img) = compute_integral_images(image.view());

        for max_matches in [1, 5, 10, 20] {
            let matches = find_similar_patches(
                image.view(),
                &sum_img,
                &sq_sum_img,
                (8, 8),
                (4, 4),
                (16, 16),
                max_matches,
                1,
            );

            assert!(
                matches.len() <= max_matches,
                "Should return at most {} matches, got {}",
                max_matches,
                matches.len()
            );
        }
    }

    #[test]
    fn test_find_similar_search_window() {
        let image = random_matrix_f32(32, 32, 99999);
        let (sum_img, sq_sum_img) = compute_integral_images(image.view());

        let ref_pos = (16, 16);
        let search_window = (8, 8);

        let matches = find_similar_patches(
            image.view(),
            &sum_img,
            &sq_sum_img,
            ref_pos,
            (4, 4),
            search_window,
            20,
            1,
        );

        for m in &matches {
            let row_dist = (m.row as i32 - ref_pos.0 as i32).abs();
            let col_dist = (m.col as i32 - ref_pos.1 as i32).abs();

            assert!(
                row_dist <= (search_window.0 / 2) as i32,
                "Match row {} outside search window of ref {}",
                m.row,
                ref_pos.0
            );
            assert!(
                col_dist <= (search_window.1 / 2) as i32,
                "Match col {} outside search window of ref {}",
                m.col,
                ref_pos.1
            );
        }
    }

    #[test]
    fn test_find_similar_boundary_patch() {
        let image = random_matrix_f32(16, 16, 11111);
        let (sum_img, sq_sum_img) = compute_integral_images(image.view());

        let matches = find_similar_patches(
            image.view(),
            &sum_img,
            &sq_sum_img,
            (0, 0),
            (4, 4),
            (8, 8),
            5,
            1,
        );
        assert!(
            !matches.is_empty(),
            "Should find matches at top-left corner"
        );

        let matches = find_similar_patches(
            image.view(),
            &sum_img,
            &sq_sum_img,
            (12, 12),
            (4, 4),
            (8, 8),
            5,
            1,
        );
        assert!(
            !matches.is_empty(),
            "Should find matches at bottom-right corner"
        );
    }

    #[test]
    fn test_find_similar_results_sorted() {
        let image = random_matrix_f32(24, 24, 77777);
        let (sum_img, sq_sum_img) = compute_integral_images(image.view());

        let matches = find_similar_patches(
            image.view(),
            &sum_img,
            &sq_sum_img,
            (8, 8),
            (4, 4),
            (12, 12),
            10,
            1,
        );

        for i in 1..matches.len() {
            assert!(
                matches[i].distance >= matches[i - 1].distance,
                "Results should be sorted: {} >= {} at index {}",
                matches[i].distance,
                matches[i - 1].distance,
                i
            );
        }
    }

    #[test]
    fn test_find_similar_with_step() {
        let image = random_matrix_f32(32, 32, 88888);
        let (sum_img, sq_sum_img) = compute_integral_images(image.view());

        let matches_step1 = find_similar_patches(
            image.view(),
            &sum_img,
            &sq_sum_img,
            (8, 8),
            (4, 4),
            (16, 16),
            50,
            1,
        );

        let matches_step2 = find_similar_patches(
            image.view(),
            &sum_img,
            &sq_sum_img,
            (8, 8),
            (4, 4),
            (16, 16),
            50,
            2,
        );

        assert!(
            matches_step2.len() <= matches_step1.len(),
            "Step=2 should find same or fewer matches than step=1: {} vs {}",
            matches_step2.len(),
            matches_step1.len()
        );
    }

    #[test]
    fn test_find_similar_minimum_image() {
        let image = random_matrix_f32(4, 4, 22222);
        let (sum_img, sq_sum_img) = compute_integral_images(image.view());

        let matches = find_similar_patches(
            image.view(),
            &sum_img,
            &sq_sum_img,
            (0, 0),
            (4, 4),
            (4, 4),
            5,
            1,
        );

        assert!(!matches.is_empty(), "Should find at least self-match");
        assert_eq!(matches[0].row, 0);
        assert_eq!(matches[0].col, 0);
        assert_eq!(matches[0].distance, 0.0);
    }

    #[test]
    fn test_find_similar_patch_equals_image() {
        let image = random_matrix_f32(8, 8, 33333);
        let (sum_img, sq_sum_img) = compute_integral_images(image.view());

        let matches = find_similar_patches(
            image.view(),
            &sum_img,
            &sq_sum_img,
            (0, 0),
            (8, 8),
            (8, 8),
            5,
            1,
        );

        assert_eq!(matches.len(), 1, "Should only find self-match");
        assert_eq!(matches[0].row, 0);
        assert_eq!(matches[0].col, 0);
        assert_eq!(matches[0].distance, 0.0);
    }

    #[test]
    fn test_compute_squared_distance_identical() {
        let image = Array2::<f32>::from_elem((8, 8), 0.5);
        let (sum_img, sq_sum_img) = compute_integral_images(image.view());

        let matches = find_similar_patches(
            image.view(),
            &sum_img,
            &sq_sum_img,
            (0, 0),
            (4, 4),
            (8, 8),
            10,
            1,
        );

        for m in &matches {
            assert!(
                m.distance < 1e-10,
                "Identical patches should have distance ~0, got {}",
                m.distance
            );
        }
    }

    #[test]
    fn test_compute_squared_distance_8x8_f32_simd_matches_scalar() {
        let image = random_matrix_f32(24, 24, 44444);
        let image_data = image
            .as_slice_memory_order()
            .expect("random_matrix_f32 should be contiguous");
        let cols = image.dim().1;
        let threshold = f32::MAX;

        for ref_r in (0..=16).step_by(3) {
            for ref_c in (0..=16).step_by(3) {
                for cand_r in (0..=16).step_by(5) {
                    for cand_c in (0..=16).step_by(5) {
                        let scalar = compute_squared_distance_at_strided_8x8_scalar(
                            image_data, cols, ref_r, ref_c, cand_r, cand_c, threshold,
                        );
                        let simd = compute_squared_distance_at_strided_8x8(
                            image_data, cols, ref_r, ref_c, cand_r, cand_c, threshold,
                        );
                        let abs_diff = (simd - scalar).abs();
                        let rel_diff = abs_diff / scalar.max(1.0);
                        assert!(
                            rel_diff < 1e-5,
                            "SIMD drift too large ref=({},{}) cand=({},{}) scalar={} simd={} rel_diff={}",
                            ref_r,
                            ref_c,
                            cand_r,
                            cand_c,
                            scalar,
                            simd,
                            rel_diff
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_find_similar_known_distance() {
        let mut image = Array2::<f32>::zeros((8, 8));
        for r in 0..4 {
            for c in 4..8 {
                image[[r, c]] = 1.0;
            }
        }
        let (sum_img, sq_sum_img) = compute_integral_images(image.view());

        let matches = find_similar_patches(
            image.view(),
            &sum_img,
            &sq_sum_img,
            (0, 0),
            (4, 4),
            (8, 8),
            5,
            1,
        );

        let distant_match = matches.iter().find(|m| m.row == 0 && m.col == 4);
        if let Some(m) = distant_match {
            assert!(
                (m.distance - 16.0).abs() < 1e-5,
                "Known distance should be 16, got {}",
                m.distance
            );
        }
    }
}
