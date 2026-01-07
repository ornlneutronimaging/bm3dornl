use ndarray::{Array2, ArrayView2, s};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatchMatch {
    pub row: usize,
    pub col: usize,
    pub distance: f32,
}

impl Eq for PatchMatch {}

impl PartialOrd for PatchMatch {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for PatchMatch {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Compute squared L2 distance between two patches with early termination.
#[inline]
fn compute_squared_distance(p1: ArrayView2<f32>, p2: ArrayView2<f32>, threshold: f32) -> f32 {
    let mut sum_sq = 0.0;
    for (r1, r2) in p1.outer_iter().zip(p2.outer_iter()) {
        for (a, b) in r1.iter().zip(r2.iter()) {
            let diff = *a - *b;
            sum_sq += diff * diff;
        }
        if sum_sq >= threshold {
            return sum_sq;
        }
    }
    sum_sq
}

/// Compute Integral Images (Sum and Squared Sum).
/// Integral Image I(x, y) = sum(i(x', y')) for x'<=x, y'<=y.
/// This allows O(1) computation of sum of pixels in any rectangular region.
/// We use this for fast pre-screening in Block Matching:
/// 1. Compute Mean Difference Bound: (sum1 - sum2)^2 / N
/// 2. Compute Norm Difference Bound: (norm1 - norm2)^2
/// If either bound exceeds threshold, we skip strict distance calculation.
/// Returns (Sum, SqSum). Indexing: [row+1, col+1].
pub fn compute_integral_images(image: ArrayView2<f32>) -> (Array2<f32>, Array2<f32>) {
    let (h, w) = image.dim();
    let mut sum_img = Array2::<f32>::zeros((h + 1, w + 1));
    let mut sq_sum_img = Array2::<f32>::zeros((h + 1, w + 1));
    
    for r in 0..h {
        let mut row_sum = 0.0;
        let mut row_sq_sum = 0.0;
        for c in 0..w {
            let val = image[[r, c]];
            row_sum += val;
            row_sq_sum += val * val;
            
            sum_img[[r + 1, c + 1]] = sum_img[[r, c + 1]] + row_sum;
            sq_sum_img[[r + 1, c + 1]] = sq_sum_img[[r, c + 1]] + row_sq_sum;
        }
    }
    (sum_img, sq_sum_img)
}

#[inline(always)]
fn get_patch_sums(
    sum_img: &Array2<f32>,
    sq_sum_img: &Array2<f32>,
    r: usize, c: usize, h: usize, w: usize
) -> (f32, f32) {
    let r1 = r;
    let c1 = c;
    let r2 = r + h;
    let c2 = c + w;
    
    let sum = sum_img[[r2, c2]] - sum_img[[r1, c2]] - sum_img[[r2, c1]] + sum_img[[r1, c1]];
    let sq_sum = sq_sum_img[[r2, c2]] - sq_sum_img[[r1, c2]] - sq_sum_img[[r2, c1]] + sq_sum_img[[r1, c1]];
    
    (sum, sq_sum)
}

/// Find similar patches within a search window using Integral Image Pre-Screening.
pub fn find_similar_patches(
    image: ArrayView2<f32>,
    integral_sum: &Array2<f32>,
    integral_sq_sum: &Array2<f32>,
    ref_pos: (usize, usize),
    patch_size: (usize, usize),
    search_window: (usize, usize),
    max_matches: usize,
    step: usize,
) -> Vec<PatchMatch> {
    let (ref_r, ref_c) = ref_pos;
    let (ph, pw) = patch_size;
    let (h, w) = image.dim();

    let ref_patch = image.slice(s![ref_r..ref_r + ph, ref_c..ref_c + pw]);
    
    let search_r_start = ref_r.saturating_sub(search_window.0 / 2);
    let search_r_end = (ref_r + search_window.0 / 2).min(h - ph);
    let search_c_start = ref_c.saturating_sub(search_window.1 / 2);
    let search_c_end = (ref_c + search_window.1 / 2).min(w - pw);

    let mut heap = std::collections::BinaryHeap::with_capacity(max_matches + 1);

    heap.push(PatchMatch { row: ref_r, col: ref_c, distance: 0.0 });

    let mut threshold = if max_matches > 1 { f32::MAX } else { 0.0 };
    
    // Pre-calculate Reference stats
    let (ref_sum, ref_sq_sum) = get_patch_sums(integral_sum, integral_sq_sum, ref_r, ref_c, ph, pw);
    let ref_norm = ref_sq_sum.max(0.0).sqrt(); 
    let inv_n = 1.0 / ((ph * pw) as f32);

    for r in (search_r_start..=search_r_end).step_by(step) {
        for c in (search_c_start..=search_c_end).step_by(step) {
             if r == ref_r && c == ref_c { continue; }

            // 1. Pre-Screening (Bounds Check)
            // Bound 1: Mean Difference: (sum1 - sum2)^2 / N
            // Bound 2: Norm Difference: (norm1 - norm2)^2
            // If Max(Bound) > threshold, skip.
            
            let (cand_sum, cand_sq_sum) = get_patch_sums(integral_sum, integral_sq_sum, r, c, ph, pw);
            let check_threshold = threshold;
            
            // Mean Bound
            let diff_sum = cand_sum - ref_sum;
            let lb_mean = (diff_sum * diff_sum) * inv_n;
            if lb_mean >= check_threshold { continue; }
            
            // Norm Bound
            let cand_norm = cand_sq_sum.max(0.0).sqrt();
            let diff_norm = (cand_norm - ref_norm).abs();
            let lb_norm = diff_norm * diff_norm;
            if lb_norm >= check_threshold { continue; }
            
            // 2. Full Distance Calculation
            let candidate_patch = image.slice(s![r..r + ph, c..c + pw]);
            let dist = compute_squared_distance(ref_patch, candidate_patch, threshold);

            if dist < threshold {
                if heap.len() < max_matches {
                    heap.push(PatchMatch { row: r, col: c, distance: dist });
                     if heap.len() == max_matches {
                        threshold = heap.peek().unwrap().distance;
                    }
                } else {
                    heap.pop();
                    heap.push(PatchMatch { row: r, col: c, distance: dist });
                    threshold = heap.peek().unwrap().distance;
                }
            }
        }
    }
    
    let mut sorted_matches = heap.into_vec();
    sorted_matches.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    
    sorted_matches
}
