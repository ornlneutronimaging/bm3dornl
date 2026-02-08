use ndarray::{Array2, ArrayView2};
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

/// Compute squared L2 distance between two patches directly from the source image.
///
/// This avoids creating candidate patch views in the hot inner loop.
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
fn compute_squared_distance_at_strided_8x8<F: Bm3dFloat>(
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

/// Compute Integral Image (sum only).
///
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

/// Compute Integral Images (Sum and Squared Sum).
///
/// Integral Image I(x, y) = sum(i(x', y')) for x'<=x, y'<=y.
/// This allows O(1) computation of sum of pixels in any rectangular region.
/// We use this for fast pre-screening in Block Matching:
/// 1. Compute Mean Difference Bound: (sum1 - sum2)^2 / N
/// 2. Compute Norm Difference Bound: (norm1 - norm2)^2
///
/// If either bound exceeds threshold, we skip strict distance calculation.
/// Returns (Sum, SqSum). Indexing: [row+1, col+1].
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
    if let (Some(image_data), Some(sum_data)) = (
        image.as_slice_memory_order(),
        integral_sum.as_slice_memory_order(),
    ) {
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
        // Fallback for non-contiguous views.
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

/// Backward-compatible in-place API.
///
/// `integral_sq_sum` is accepted for compatibility but no longer used.
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

    // Helper: Simple Linear Congruential Generator for deterministic "random" test data
    // Copied from transforms.rs tests to maintain consistency
    struct SimpleLcg {
        state: u64,
    }

    impl SimpleLcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            // LCG parameters from Numerical Recipes
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.state
        }

        fn next_f32(&mut self) -> f32 {
            // Generate f32 in range [0.0, 1.0)
            let u = self.next_u64();
            (u >> 40) as f32 / (1u64 << 24) as f32
        }

        fn next_f64(&mut self) -> f64 {
            // Generate f64 in range [0.0, 1.0)
            let u = self.next_u64();
            (u >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    // Helper: Generate deterministic "random" matrix
    fn random_matrix_f32(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
        let mut rng = SimpleLcg::new(seed);
        Array2::from_shape_fn((rows, cols), |_| rng.next_f32())
    }

    fn random_matrix_f64(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
        let mut rng = SimpleLcg::new(seed);
        Array2::from_shape_fn((rows, cols), |_| rng.next_f64())
    }

    // ==================== PatchMatch Struct Tests ====================

    #[test]
    fn test_patch_match_ordering_by_distance() {
        // Verify Ord impl sorts by distance ascending (for min-heap behavior with BinaryHeap)
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

        // Ord comparison: smaller distance should be Less
        assert!(p3 < p1, "0.5 should be less than 1.0");
        assert!(p1 < p2, "1.0 should be less than 2.0");
        assert!(p3 < p2, "0.5 should be less than 2.0");
    }

    #[test]
    fn test_patch_match_heap_behavior() {
        // BinaryHeap is a max-heap, so largest distance pops first
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

        // Pop order should be: 3.0, 2.0, 1.0, 0.5 (max-heap)
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
        // When distances are equal, Ord should return Equal
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
        // Verify heap handles equal distances without panic
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

        // All should pop with distance 1.0
        for _ in 0..3 {
            let p = heap.pop().unwrap();
            assert_eq!(p.distance, 1.0);
        }
    }

    // ==================== Integral Image Tests ====================

    #[test]
    fn test_integral_image_simple() {
        // Simple 2x2 matrix with known values
        // [1, 2]
        // [3, 4]
        let mut input = Array2::<f32>::zeros((2, 2));
        input[[0, 0]] = 1.0;
        input[[0, 1]] = 2.0;
        input[[1, 0]] = 3.0;
        input[[1, 1]] = 4.0;

        let (sum_img, sq_sum_img) = compute_integral_images(input.view());

        // Sum integral image (3x3 with zero padding):
        // [0, 0, 0]
        // [0, 1, 3]     (1, 1+2=3)
        // [0, 4, 10]    (1+3=4, 1+2+3+4=10)
        assert_eq!(sum_img.dim(), (3, 3));
        assert_eq!(sum_img[[0, 0]], 0.0);
        assert_eq!(sum_img[[0, 1]], 0.0);
        assert_eq!(sum_img[[0, 2]], 0.0);
        assert_eq!(sum_img[[1, 0]], 0.0);
        assert_eq!(sum_img[[1, 1]], 1.0);
        assert_eq!(sum_img[[1, 2]], 3.0); // 1 + 2
        assert_eq!(sum_img[[2, 0]], 0.0);
        assert_eq!(sum_img[[2, 1]], 4.0); // 1 + 3
        assert_eq!(sum_img[[2, 2]], 10.0); // 1 + 2 + 3 + 4

        // Squared sum integral image:
        // [0, 0, 0]
        // [0, 1, 5]     (1, 1+4=5)
        // [0, 10, 30]   (1+9=10, 1+4+9+16=30)
        assert_eq!(sq_sum_img[[1, 1]], 1.0);
        assert_eq!(sq_sum_img[[1, 2]], 5.0); // 1 + 4
        assert_eq!(sq_sum_img[[2, 1]], 10.0); // 1 + 9
        assert_eq!(sq_sum_img[[2, 2]], 30.0); // 1 + 4 + 9 + 16
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

        // For all-ones, integral at (i+1, j+1) should equal (i+1) * (j+1)
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
        // Non-square 2x3 matrix
        // [1, 2, 3]
        // [4, 5, 6]
        let mut input = Array2::<f32>::zeros((2, 3));
        for r in 0..2 {
            for c in 0..3 {
                input[[r, c]] = (r * 3 + c + 1) as f32;
            }
        }

        let (sum_img, _) = compute_integral_images(input.view());

        assert_eq!(sum_img.dim(), (3, 4));
        // Full sum should be 1+2+3+4+5+6 = 21
        assert_eq!(sum_img[[2, 3]], 21.0);
        // First row sum: 1+2+3 = 6
        assert_eq!(sum_img[[1, 3]], 6.0);
        // First column sum: 1+4 = 5
        assert_eq!(sum_img[[2, 1]], 5.0);
    }

    #[test]
    fn test_integral_image_large_values() {
        // Test numerical stability with large values ~1e6
        let mut input = Array2::<f32>::zeros((4, 4));
        for r in 0..4 {
            for c in 0..4 {
                input[[r, c]] = 1e6;
            }
        }

        let (sum_img, sq_sum_img) = compute_integral_images(input.view());

        // Total sum should be 16 * 1e6 = 1.6e7
        let expected_sum = 16.0 * 1e6;
        assert!(
            (sum_img[[4, 4]] - expected_sum).abs() < 1.0,
            "Large value sum should be correct: got {}, expected {}",
            sum_img[[4, 4]],
            expected_sum
        );

        // Total squared sum should be 16 * (1e6)^2 = 1.6e13
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

    // ==================== find_similar_patches Tests ====================

    #[test]
    fn test_find_similar_identical_image() {
        // Uniform image (all same value) - all patches equally similar
        let image = Array2::<f32>::from_elem((16, 16), 0.5);
        let (sum_img, sq_sum_img) = compute_integral_images(image.view());

        let matches = find_similar_patches(
            image.view(),
            &sum_img,
            &sq_sum_img,
            (4, 4), // ref_pos
            (4, 4), // patch_size
            (8, 8), // search_window
            5,      // max_matches
            1,      // step
        );

        // All matches should have distance ≈ 0
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
        // Reference patch should always be included with distance ≈ 0
        let image = random_matrix_f32(16, 16, 12345);
        let (sum_img, sq_sum_img) = compute_integral_images(image.view());

        let matches = find_similar_patches(
            image.view(),
            &sum_img,
            &sq_sum_img,
            (4, 4), // ref_pos
            (4, 4), // patch_size
            (8, 8), // search_window
            10,     // max_matches
            1,      // step
        );

        // Should contain self-match at (4, 4) with distance 0
        let self_match = matches.iter().find(|m| m.row == 4 && m.col == 4);
        assert!(self_match.is_some(), "Self-match should be in results");
        assert_eq!(
            self_match.unwrap().distance,
            0.0,
            "Self-match distance should be 0"
        );

        // Self-match should be first (smallest distance)
        assert_eq!(matches[0].row, 4);
        assert_eq!(matches[0].col, 4);
        assert_eq!(matches[0].distance, 0.0);
    }

    #[test]
    fn test_find_similar_f64() {
        // Test with f64 precision
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

        // Self-match should be at distance 0
        assert_eq!(matches[0].row, 4);
        assert_eq!(matches[0].col, 4);
        assert_eq!(matches[0].distance, 0.0);
    }

    #[test]
    fn test_find_similar_distinct_regions() {
        // Image with two distinct constant regions
        // Left half = 0.0, Right half = 1.0
        let mut image = Array2::<f32>::zeros((16, 16));
        for r in 0..16 {
            for c in 8..16 {
                image[[r, c]] = 1.0;
            }
        }
        let (sum_img, sq_sum_img) = compute_integral_images(image.view());

        // Reference patch in left region (all zeros)
        let matches = find_similar_patches(
            image.view(),
            &sum_img,
            &sq_sum_img,
            (4, 2),   // ref_pos (in left region)
            (4, 4),   // patch_size
            (16, 16), // large search window
            5,        // max_matches
            1,        // step
        );

        // All best matches should be in left region (columns 0-4 for 4-wide patches)
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
        // Large image, small search window - only finds patches within window
        let image = random_matrix_f32(32, 32, 99999);
        let (sum_img, sq_sum_img) = compute_integral_images(image.view());

        let ref_pos = (16, 16);
        let search_window = (8, 8); // ±4 from reference

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

        // All matches should be within search window of reference
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
        // Reference patch at image edge - should not panic
        let image = random_matrix_f32(16, 16, 11111);
        let (sum_img, sq_sum_img) = compute_integral_images(image.view());

        // Top-left corner
        let matches = find_similar_patches(
            image.view(),
            &sum_img,
            &sq_sum_img,
            (0, 0), // corner position
            (4, 4),
            (8, 8),
            5,
            1,
        );
        assert!(
            !matches.is_empty(),
            "Should find matches at top-left corner"
        );

        // Bottom-right corner (need to leave room for patch)
        let matches = find_similar_patches(
            image.view(),
            &sum_img,
            &sq_sum_img,
            (12, 12), // 16-4=12, last valid position for 4x4 patch
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
        // Verify results are sorted by distance (ascending)
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
        // Verify step parameter affects search
        let image = random_matrix_f32(32, 32, 88888);
        let (sum_img, sq_sum_img) = compute_integral_images(image.view());

        let matches_step1 = find_similar_patches(
            image.view(),
            &sum_img,
            &sq_sum_img,
            (8, 8),
            (4, 4),
            (16, 16),
            50, // Request many matches
            1,  // step=1
        );

        let matches_step2 = find_similar_patches(
            image.view(),
            &sum_img,
            &sq_sum_img,
            (8, 8),
            (4, 4),
            (16, 16),
            50,
            2, // step=2
        );

        // With step=2, we search fewer positions, so typically get fewer or equal matches
        // (assuming enough positions exist)
        assert!(
            matches_step2.len() <= matches_step1.len(),
            "Step=2 should find same or fewer matches than step=1: {} vs {}",
            matches_step2.len(),
            matches_step1.len()
        );
    }

    // ==================== Edge Case Tests ====================

    #[test]
    fn test_find_similar_minimum_image() {
        // Smallest valid image for a 4x4 patch
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

        // Should return at least self-match
        assert!(!matches.is_empty(), "Should find at least self-match");
        assert_eq!(matches[0].row, 0);
        assert_eq!(matches[0].col, 0);
        assert_eq!(matches[0].distance, 0.0);
    }

    #[test]
    fn test_find_similar_patch_equals_image() {
        // Patch size equals image size - only one possible patch (self)
        let image = random_matrix_f32(8, 8, 33333);
        let (sum_img, sq_sum_img) = compute_integral_images(image.view());

        let matches = find_similar_patches(
            image.view(),
            &sum_img,
            &sq_sum_img,
            (0, 0),
            (8, 8), // patch_size equals image
            (8, 8),
            5,
            1,
        );

        // Only self-match possible
        assert_eq!(matches.len(), 1, "Should only find self-match");
        assert_eq!(matches[0].row, 0);
        assert_eq!(matches[0].col, 0);
        assert_eq!(matches[0].distance, 0.0);
    }

    #[test]
    fn test_compute_squared_distance_identical() {
        // Test the internal distance function via find_similar_patches
        // Two identical patches should have distance 0
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

        // All patches identical, all distances should be 0
        for m in &matches {
            assert!(
                m.distance < 1e-10,
                "Identical patches should have distance ~0, got {}",
                m.distance
            );
        }
    }

    #[test]
    fn test_find_similar_known_distance() {
        // Create image where we can predict exact distances
        // Reference patch at (0,0) with value 0, another patch at (0,4) with value 1
        // Distance = 4*4 * (1-0)^2 = 16
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

        // Find the match at (0, 4)
        let distant_match = matches.iter().find(|m| m.row == 0 && m.col == 4);
        if let Some(m) = distant_match {
            // Distance should be 16 (4*4 pixels, each diff=1, squared=1)
            assert!(
                (m.distance - 16.0).abs() < 1e-5,
                "Known distance should be 16, got {}",
                m.distance
            );
        }
    }
}
