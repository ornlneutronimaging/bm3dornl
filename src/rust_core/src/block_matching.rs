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

/// Compute squared L2 distance between two patches.
/// Assumes patches have the same shape.
#[inline]
fn compute_squared_distance(p1: ArrayView2<f32>, p2: ArrayView2<f32>) -> f32 {
    let mut sum_sq = 0.0;
    // Iterating with zip is usually efficient in ndarray
    for (a, b) in p1.iter().zip(p2.iter()) {
        let diff = *a - *b;
        sum_sq += diff * diff;
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
    // Note: Caller must ensure ref_pos is valid.
    let ref_patch = image.slice(s![ref_r..ref_r + ph, ref_c..ref_c + pw]);

    // Define search bounds
    let search_r_start = ref_r.saturating_sub(search_window.0 / 2);
    let search_r_end = (ref_r + search_window.0 / 2).min(h - ph);
    let search_c_start = ref_c.saturating_sub(search_window.1 / 2);
    let search_c_end = (ref_c + search_window.1 / 2).min(w - pw);

    let mut matches = Vec::with_capacity(max_matches);

    // Initial match (self)
    matches.push(PatchMatch {
        row: ref_r,
        col: ref_c,
        distance: 0.0,
    });

    for r in (search_r_start..=search_r_end).step_by(step) {
        for c in (search_c_start..=search_c_end).step_by(step) {
            if r == ref_r && c == ref_c {
                continue;
            }

            let candidate_patch = image.slice(s![r..r + ph, c..c + pw]);
            let dist = compute_squared_distance(ref_patch, candidate_patch);

            // Simple selection sort logic/insertion logic
            // Since max_matches is small (e.g., 16 or 32), a simple linear scan or
            // just collecting all and sorting partially might be fine.
            // But strict "keep top K" usually implies a heap or bounded buffer.
            
            // Optimization: if we already have K matches, check if this is smaller than the worst one.
             matches.push(PatchMatch {
                row: r,
                col: c,
                distance: dist,
            });

        }
    }
    
    // Sort by distance
    matches.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    
    // Keep top K
    if matches.len() > max_matches {
        matches.truncate(max_matches);
    }

    matches
}
