use ndarray::{ArrayView2, s};
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
        // We want min-heap behavior for keeping the smallest distances.
        // But standard BinaryHeap is a max-heap.
        // So we reverse the comparison: bigger distance = "smaller" element (to be popped).
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for PatchMatch {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Compute squared L2 distance between two patches with early termination.
/// Assumes patches have the same shape.
/// optimized to check threshold only after each row to allow inner loop vectorization.
#[inline]
fn compute_squared_distance(p1: ArrayView2<f32>, p2: ArrayView2<f32>, threshold: f32) -> f32 {
    let mut sum_sq = 0.0;
    // Iterate rows
    for (r1, r2) in p1.outer_iter().zip(p2.outer_iter()) {
        // Compute row squared difference
        for (a, b) in r1.iter().zip(r2.iter()) {
            let diff = *a - *b;
            sum_sq += diff * diff;
        }
        
        // Check threshold after entire row
        if sum_sq >= threshold {
            return sum_sq;
        }
    }
    sum_sq
}

/// Find similar patches within a search window.
///
/// # Arguments
/// * `image` - The full image (2D array).
/// * `ref_pos` - (row, col) of the top-left corner of the reference patch.
/// * `patch_size` - Size of the patch (height, width).
/// * `search_window` - (height, width) of the search area centered on ref_pos.
/// * `max_matches` - Maximum number of similar patches to keep (N_hard).
pub fn find_similar_patches(
    image: ArrayView2<f32>,
    ref_pos: (usize, usize),
    patch_size: (usize, usize),
    search_window: (usize, usize),
    max_matches: usize,
    step: usize,
) -> Vec<PatchMatch> {
    let (ref_r, ref_c) = ref_pos;
    let (ph, pw) = patch_size;
    let (h, w) = image.dim();

    // Extract reference patch
    let ref_patch = image.slice(s![ref_r..ref_r + ph, ref_c..ref_c + pw]);

    // Define search bounds
    let search_r_start = ref_r.saturating_sub(search_window.0 / 2);
    let search_r_end = (ref_r + search_window.0 / 2).min(h - ph);
    let search_c_start = ref_c.saturating_sub(search_window.1 / 2);
    let search_c_end = (ref_c + search_window.1 / 2).min(w - pw);

    let mut heap = std::collections::BinaryHeap::with_capacity(max_matches + 1);

    // Initial match (self)
    heap.push(PatchMatch {
        row: ref_r,
        col: ref_c,
        distance: 0.0, 
    });

    // Current worst distance in the top-K. 
    // If heap is not full, acceptance threshold is effectively infinite.
    let mut threshold = if max_matches > 1 { f32::MAX } else { 0.0 };

    for r in (search_r_start..=search_r_end).step_by(step) {
        for c in (search_c_start..=search_c_end).step_by(step) {
             if r == ref_r && c == ref_c {
                continue;
            }

            let candidate_patch = image.slice(s![r..r + ph, c..c + pw]);
            
            // Optimization: Early termination if distance exceeds current worst in heap
            let dist = compute_squared_distance(ref_patch, candidate_patch, threshold);

            if dist < threshold {
                if heap.len() < max_matches {
                    heap.push(PatchMatch { row: r, col: c, distance: dist });
                     if heap.len() == max_matches {
                        // Heap just filled up. Set threshold to the worst element.
                        threshold = heap.peek().unwrap().distance;
                    }
                } else {
                    // Heap is full, replace the worst
                    heap.pop();
                    heap.push(PatchMatch { row: r, col: c, distance: dist });
                    threshold = heap.peek().unwrap().distance;
                }
            }
        }
    }
    
    // Convert to sorted vector
    let mut sorted_matches = heap.into_vec();
    sorted_matches.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    
    sorted_matches
}
