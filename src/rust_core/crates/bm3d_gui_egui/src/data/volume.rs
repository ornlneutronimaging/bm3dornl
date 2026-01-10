use ndarray::{Array2, Array3, Axis};

/// Axis mapping configuration for 3D volume display.
/// Maps volume dimensions to display axes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxisMapping {
    /// Dimension index for horizontal (X) axis
    pub x_axis: usize,
    /// Dimension index for vertical (Y) axis
    pub y_axis: usize,
    /// Dimension index for slice navigation
    pub slice_axis: usize,
}

impl Default for AxisMapping {
    /// Default: X=D2, Y=D0, Slice=D1 (ORNL sinogram standard)
    fn default() -> Self {
        Self {
            x_axis: 2,
            y_axis: 0,
            slice_axis: 1,
        }
    }
}

impl AxisMapping {
    /// Check if the mapping is valid (all axes different).
    pub fn is_valid(&self) -> bool {
        self.x_axis != self.y_axis
            && self.y_axis != self.slice_axis
            && self.x_axis != self.slice_axis
            && self.x_axis < 3
            && self.y_axis < 3
            && self.slice_axis < 3
    }

    /// Get dimension labels for UI display.
    pub fn dimension_labels() -> [&'static str; 3] {
        ["D0", "D1", "D2"]
    }
}

/// 3D volume data with axis mapping support.
pub struct Volume3D {
    /// Raw data stored as f32, shape is [D0, D1, D2]
    data: Array3<f32>,
    /// Current axis mapping
    axis_mapping: AxisMapping,
    /// Original shape before any permutation
    original_shape: [usize; 3],
}

impl Volume3D {
    /// Create a new Volume3D from raw data.
    pub fn new(data: Array3<f32>) -> Self {
        let shape = data.shape();
        let original_shape = [shape[0], shape[1], shape[2]];
        Self {
            data,
            axis_mapping: AxisMapping::default(),
            original_shape,
        }
    }

    /// Get the original shape [D0, D1, D2].
    pub fn original_shape(&self) -> [usize; 3] {
        self.original_shape
    }

    /// Get current axis mapping.
    pub fn axis_mapping(&self) -> AxisMapping {
        self.axis_mapping
    }

    /// Get reference to raw data array.
    pub fn raw_data(&self) -> &Array3<f32> {
        &self.data
    }

    /// Set new axis mapping.
    pub fn set_axis_mapping(&mut self, mapping: AxisMapping) {
        if mapping.is_valid() {
            self.axis_mapping = mapping;
        }
    }

    /// Get the number of slices along the current slice axis.
    pub fn num_slices(&self) -> usize {
        self.original_shape[self.axis_mapping.slice_axis]
    }

    /// Get the display dimensions (width, height) for current mapping.
    pub fn display_dimensions(&self) -> (usize, usize) {
        let width = self.original_shape[self.axis_mapping.x_axis];
        let height = self.original_shape[self.axis_mapping.y_axis];
        (width, height)
    }

    /// Extract a 2D slice at the given index along the slice axis.
    /// Returns data ready for display with shape (height, width).
    pub fn get_slice(&self, slice_index: usize) -> Option<Array2<f32>> {
        let num_slices = self.num_slices();
        if slice_index >= num_slices {
            return None;
        }

        let slice_axis = self.axis_mapping.slice_axis;
        let x_axis = self.axis_mapping.x_axis;
        let y_axis = self.axis_mapping.y_axis;

        // Take slice along slice_axis
        let slice_2d = self.data.index_axis(Axis(slice_axis), slice_index);

        // Determine the remaining axes after removing slice_axis
        // We need to map original axes to the 2D slice axes
        let remaining_axes: Vec<usize> = (0..3).filter(|&a| a != slice_axis).collect();

        // Find where x_axis and y_axis are in the remaining axes
        let x_pos = remaining_axes.iter().position(|&a| a == x_axis).unwrap();
        let y_pos = remaining_axes.iter().position(|&a| a == y_axis).unwrap();

        // slice_2d has shape corresponding to remaining_axes order
        // We need output with shape (height=y_axis_size, width=x_axis_size)
        let slice_owned = slice_2d.to_owned();

        // If y_pos < x_pos, shape is already (y, x) - good
        // If y_pos > x_pos, shape is (x, y) - need to transpose
        if y_pos > x_pos {
            Some(slice_owned.reversed_axes())
        } else {
            Some(slice_owned)
        }
    }

    /// Convert a slice to grayscale u8 pixels, auto-scaled to min/max.
    pub fn slice_to_grayscale(&self, slice_index: usize) -> Option<(Vec<u8>, usize, usize)> {
        let slice = self.get_slice(slice_index)?;
        let (height, width) = (slice.nrows(), slice.ncols());

        // Find min/max for auto-scaling
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &val in slice.iter() {
            if val.is_finite() {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }

        let range = if (max_val - min_val).abs() < f32::EPSILON {
            1.0
        } else {
            max_val - min_val
        };

        // Convert to u8 grayscale
        let pixels: Vec<u8> = slice
            .iter()
            .map(|&val| {
                let normalized = (val - min_val) / range;
                (normalized.clamp(0.0, 1.0) * 255.0) as u8
            })
            .collect();

        Some((pixels, width, height))
    }
}
