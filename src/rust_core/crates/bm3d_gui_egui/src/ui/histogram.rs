use crate::ui::slice_view::ImageRoi;
use eframe::egui;
use egui_plot::{Bar, BarChart, Corner, Legend, Plot};
use ndarray::Array2;

/// Theme-aware color palette for consistent visibility in light and dark modes.
/// Returns appropriate colors based on the current UI theme.
pub mod colors {
    use eframe::egui;

    /// Get ROI indicator color (for text labels showing ROI coordinates).
    /// Uses amber/gold tones that work in both themes.
    pub fn roi_label(ui: &egui::Ui, hovering: bool) -> egui::Color32 {
        let dark_mode = ui.visuals().dark_mode;
        if hovering {
            // Hovering: brighter/more saturated
            if dark_mode {
                egui::Color32::from_rgb(255, 180, 50) // Bright gold
            } else {
                egui::Color32::from_rgb(180, 100, 0) // Dark amber (readable on white)
            }
        } else {
            // Normal: slightly muted
            if dark_mode {
                egui::Color32::from_rgb(230, 160, 40) // Gold
            } else {
                egui::Color32::from_rgb(160, 90, 0) // Brown/amber (readable on white)
            }
        }
    }

    /// Get original data color (for histograms, labels).
    pub fn original(ui: &egui::Ui) -> egui::Color32 {
        if ui.visuals().dark_mode {
            egui::Color32::from_rgb(100, 149, 237) // Cornflower blue
        } else {
            egui::Color32::from_rgb(30, 80, 180) // Darker blue for light mode
        }
    }

    /// Get processed data color (for histograms, labels).
    pub fn processed(ui: &egui::Ui) -> egui::Color32 {
        if ui.visuals().dark_mode {
            egui::Color32::from_rgb(255, 140, 0) // Dark orange
        } else {
            egui::Color32::from_rgb(200, 80, 0) // Darker orange for light mode
        }
    }

    /// Get difference data color (for histograms, labels).
    pub fn difference(ui: &egui::Ui) -> egui::Color32 {
        if ui.visuals().dark_mode {
            egui::Color32::from_rgb(220, 80, 80) // Lighter crimson
        } else {
            egui::Color32::from_rgb(180, 20, 40) // Darker crimson for light mode
        }
    }

    /// Get histogram bar color with transparency.
    pub fn histogram_bar(ui: &egui::Ui, alpha: u8) -> egui::Color32 {
        let base = original(ui);
        egui::Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), alpha)
    }

    /// Get original histogram bar color with transparency.
    pub fn original_bar(ui: &egui::Ui, alpha: u8) -> egui::Color32 {
        let base = original(ui);
        egui::Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), alpha)
    }

    /// Get processed histogram bar color with transparency.
    pub fn processed_bar(ui: &egui::Ui, alpha: u8) -> egui::Color32 {
        let base = processed(ui);
        egui::Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), alpha)
    }

    /// Get difference histogram bar color with transparency.
    pub fn difference_bar(ui: &egui::Ui, alpha: u8) -> egui::Color32 {
        let base = difference(ui);
        egui::Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), alpha)
    }
}

/// Statistics computed from image data.
#[derive(Debug, Clone, Copy, Default)]
pub struct ImageStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
    pub pixel_count: usize,
}

impl ImageStats {
    pub fn compute(data: &Array2<f32>) -> Self {
        Self::compute_with_roi(data, None)
    }

    /// Compute statistics for the full image or an ROI subset.
    pub fn compute_with_roi(data: &Array2<f32>, roi: Option<&ImageRoi>) -> Self {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut count = 0usize;

        // Iterate over ROI or full image
        let (row_range, col_range) = if let Some(r) = roi {
            let row_end = (r.max_y + 1).min(data.nrows());
            let col_end = (r.max_x + 1).min(data.ncols());
            (r.min_y..row_end, r.min_x..col_end)
        } else {
            (0..data.nrows(), 0..data.ncols())
        };

        for row in row_range.clone() {
            for col in col_range.clone() {
                let val = data[[row, col]];
                if val.is_finite() {
                    min = min.min(val);
                    max = max.max(val);
                    sum += val as f64;
                    count += 1;
                }
            }
        }

        let mean = if count > 0 {
            (sum / count as f64) as f32
        } else {
            0.0
        };

        // Compute standard deviation
        let mut variance_sum = 0.0f64;
        for row in row_range {
            for col in col_range.clone() {
                let val = data[[row, col]];
                if val.is_finite() {
                    let diff = val as f64 - mean as f64;
                    variance_sum += diff * diff;
                }
            }
        }
        let std = if count > 1 {
            ((variance_sum / (count - 1) as f64).sqrt()) as f32
        } else {
            0.0
        };

        Self { min, max, mean, std, pixel_count: count }
    }
}

/// Histogram data with bin counts and range.
#[derive(Debug, Clone)]
pub struct HistogramData {
    /// Normalized bin counts (0 to 1)
    pub counts: Vec<f32>,
    /// Value at start of first bin
    pub min_val: f32,
    /// Value at end of last bin
    pub max_val: f32,
    /// Image statistics
    pub stats: ImageStats,
}

impl HistogramData {
    /// Compute histogram from image data with specified number of bins.
    pub fn compute(data: &Array2<f32>, num_bins: usize) -> Self {
        Self::compute_with_roi(data, num_bins, None)
    }

    /// Compute histogram from image data with specified number of bins, optionally limited to an ROI.
    pub fn compute_with_roi(data: &Array2<f32>, num_bins: usize, roi: Option<&ImageRoi>) -> Self {
        let stats = ImageStats::compute_with_roi(data, roi);

        // Handle edge case where all values are the same
        let range = stats.max - stats.min;
        if range < f32::EPSILON || num_bins == 0 {
            return Self {
                counts: vec![1.0],
                min_val: stats.min,
                max_val: stats.max,
                stats,
            };
        }

        // Compute bin counts
        let mut counts = vec![0usize; num_bins];
        let bin_width = range / num_bins as f32;

        // Iterate over ROI or full image
        let (row_range, col_range) = if let Some(r) = roi {
            let row_end = (r.max_y + 1).min(data.nrows());
            let col_end = (r.max_x + 1).min(data.ncols());
            (r.min_y..row_end, r.min_x..col_end)
        } else {
            (0..data.nrows(), 0..data.ncols())
        };

        for row in row_range {
            for col in col_range.clone() {
                let val = data[[row, col]];
                if val.is_finite() {
                    let bin_idx = ((val - stats.min) / bin_width) as usize;
                    // Clamp to valid range (handle edge case where val == max)
                    let bin_idx = bin_idx.min(num_bins - 1);
                    counts[bin_idx] += 1;
                }
            }
        }

        // Find max count for normalization
        let max_count = counts.iter().copied().max().unwrap_or(1) as f32;

        // Normalize counts to [0, 1]
        let normalized: Vec<f32> = counts
            .iter()
            .map(|&c| c as f32 / max_count)
            .collect();

        Self {
            counts: normalized,
            min_val: stats.min,
            max_val: stats.max,
            stats,
        }
    }
}

/// State for histogram display in single view mode.
#[derive(Debug, Clone, Default)]
pub struct SingleViewHistogram {
    pub show: bool,
    cached_slice_index: Option<usize>,
    cached_roi: Option<ImageRoi>,
    cached_histogram: Option<HistogramData>,
}

impl SingleViewHistogram {
    pub fn new() -> Self {
        Self {
            show: false,
            cached_slice_index: None,
            cached_roi: None,
            cached_histogram: None,
        }
    }

    /// Reset cached data (call when loading new volume).
    pub fn reset(&mut self) {
        self.cached_slice_index = None;
        self.cached_roi = None;
        self.cached_histogram = None;
    }

    /// Show histogram panel. Returns the height used.
    pub fn show(&mut self, ui: &mut egui::Ui, slice_data: Option<&Array2<f32>>, slice_index: usize) -> f32 {
        self.show_with_roi(ui, slice_data, slice_index, None, &mut false, &mut false, false)
    }

    /// Show histogram panel with optional ROI. Returns the height used.
    /// Also provides:
    /// - `clear_roi_clicked`: set to true when Clear ROI button is clicked
    /// - `roi_mode`: mutable reference to ROI drawing mode toggle
    /// - `cursor_in_roi`: whether cursor is currently inside the ROI (for UI feedback)
    pub fn show_with_roi(
        &mut self,
        ui: &mut egui::Ui,
        slice_data: Option<&Array2<f32>>,
        slice_index: usize,
        roi: Option<&ImageRoi>,
        clear_roi_clicked: &mut bool,
        roi_mode: &mut bool,
        cursor_in_roi: bool,
    ) -> f32 {
        let mut height_used = 0.0;

        // Top row: histogram toggle, ROI mode button, and ROI info
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show, "Show Histogram")
                .on_hover_text("Toggle histogram display for current slice");

            ui.separator();

            // ROI mode toggle button
            let roi_button_text = if *roi_mode { "📐 ROI Mode ✓" } else { "📐 ROI Mode" };
            let roi_button = ui.selectable_label(*roi_mode, roi_button_text);
            if roi_button.clicked() {
                *roi_mode = !*roi_mode;
            }
            roi_button.on_hover_text(
                "Click to toggle ROI selection mode.\n\
                 When ON: drag on image draws ROI.\n\
                 When OFF: drag pans image.\n\
                 Shortcut: Shift+drag always draws ROI.\n\
                 Drag inside existing ROI to move it."
            );

            if let Some(r) = roi {
                ui.separator();
                // Show cursor-in-roi indicator with theme-aware colors
                let roi_text = if cursor_in_roi {
                    format!("ROI: ({},{})→({},{}) [drag to move]", r.min_x, r.min_y, r.max_x, r.max_y)
                } else {
                    format!("ROI: ({},{})→({},{})", r.min_x, r.min_y, r.max_x, r.max_y)
                };
                let roi_color = colors::roi_label(ui, cursor_in_roi);
                ui.label(
                    egui::RichText::new(roi_text)
                        .small()
                        .color(roi_color),
                );
                if ui.small_button("Clear")
                    .on_hover_text("Remove ROI selection")
                    .clicked()
                {
                    *clear_roi_clicked = true;
                }
            }
        });
        height_used += 20.0;

        if !self.show {
            return height_used;
        }

        // Update cached histogram if slice changed or ROI changed
        let roi_changed = self.cached_roi != roi.copied();
        if let Some(data) = slice_data {
            if self.cached_slice_index != Some(slice_index) || roi_changed || self.cached_histogram.is_none() {
                self.cached_histogram = Some(HistogramData::compute_with_roi(data, 256, roi));
                self.cached_slice_index = Some(slice_index);
                self.cached_roi = roi.copied();
            }
        }

        if let Some(histogram) = &self.cached_histogram {
            let plot_height = 100.0;
            height_used += plot_height + 25.0; // Plot + stats line

            // Get theme-aware bar color
            let bar_color = colors::histogram_bar(ui, 180);

            // Draw histogram plot
            let bars: Vec<Bar> = histogram
                .counts
                .iter()
                .enumerate()
                .map(|(i, &count)| {
                    let bin_width = (histogram.max_val - histogram.min_val) / histogram.counts.len() as f32;
                    let x = histogram.min_val + (i as f32 + 0.5) * bin_width;
                    Bar::new(x as f64, count as f64)
                        .width(bin_width as f64 * 0.9)
                })
                .collect();

            let chart_name = if roi.is_some() { "ROI Intensity" } else { "Intensity" };
            let chart = BarChart::new(bars)
                .color(bar_color)
                .name(chart_name);

            Plot::new("single_histogram")
                .height(plot_height)
                .show_axes([true, true])
                .show_grid(true)
                .allow_drag(false)
                .allow_zoom(false)
                .allow_scroll(false)
                .allow_boxed_zoom(false)
                .x_axis_label("Intensity")
                .y_axis_label("Count")
                .show(ui, |plot_ui| {
                    plot_ui.bar_chart(chart);
                });

            // Statistics line
            let stats = &histogram.stats;
            ui.horizontal(|ui| {
                let stats_text = if roi.is_some() {
                    format!(
                        "Pixels: {}  min: {:.3}  max: {:.3}  μ: {:.3}  σ: {:.3}",
                        stats.pixel_count, stats.min, stats.max, stats.mean, stats.std
                    )
                } else {
                    format!(
                        "min: {:.3}  max: {:.3}  μ: {:.3}  σ: {:.3}",
                        stats.min, stats.max, stats.mean, stats.std
                    )
                };
                ui.label(
                    egui::RichText::new(stats_text)
                        .small()
                        .weak(),
                );
            });
        }

        height_used
    }
}

/// State for histogram display in comparison view mode.
#[derive(Debug, Clone, Default)]
pub struct CompareViewHistogram {
    pub show: bool,
    pub show_original: bool,
    pub show_processed: bool,
    pub show_difference: bool,
    cached_slice_index: Option<usize>,
    cached_roi: Option<ImageRoi>,
    cached_original: Option<HistogramData>,
    cached_processed: Option<HistogramData>,
    cached_difference: Option<HistogramData>,
}

impl CompareViewHistogram {
    pub fn new() -> Self {
        Self {
            show: false,
            show_original: true,
            show_processed: true,
            show_difference: false,
            cached_slice_index: None,
            cached_roi: None,
            cached_original: None,
            cached_processed: None,
            cached_difference: None,
        }
    }

    /// Reset cached data.
    pub fn reset(&mut self) {
        self.cached_slice_index = None;
        self.cached_roi = None;
        self.cached_original = None;
        self.cached_processed = None;
        self.cached_difference = None;
    }

    /// Show histogram panel for comparison view.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        original_slice: Option<&Array2<f32>>,
        processed_slice: Option<&Array2<f32>>,
        slice_index: usize,
    ) -> f32 {
        self.show_with_roi(ui, original_slice, processed_slice, slice_index, None, &mut false, &mut false, false)
    }

    /// Show histogram panel for comparison view with optional ROI.
    /// Also provides:
    /// - `clear_roi_clicked`: set to true when Clear ROI button is clicked
    /// - `roi_mode`: mutable reference to ROI drawing mode toggle
    /// - `cursor_in_roi`: whether cursor is currently inside the ROI (for UI feedback)
    pub fn show_with_roi(
        &mut self,
        ui: &mut egui::Ui,
        original_slice: Option<&Array2<f32>>,
        processed_slice: Option<&Array2<f32>>,
        slice_index: usize,
        roi: Option<&ImageRoi>,
        clear_roi_clicked: &mut bool,
        roi_mode: &mut bool,
        cursor_in_roi: bool,
    ) -> f32 {
        let mut height_used = 0.0;

        // Toggle controls
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show, "Show Histogram")
                .on_hover_text("Toggle histogram display");

            if self.show {
                ui.separator();
                ui.checkbox(&mut self.show_original, "Original")
                    .on_hover_text("Show original image histogram");
                ui.checkbox(&mut self.show_processed, "Processed")
                    .on_hover_text("Show processed image histogram");
                ui.checkbox(&mut self.show_difference, "Diff")
                    .on_hover_text("Show difference histogram");
            }

            ui.separator();

            // ROI mode toggle button
            let roi_button_text = if *roi_mode { "📐 ROI Mode ✓" } else { "📐 ROI Mode" };
            let roi_button = ui.selectable_label(*roi_mode, roi_button_text);
            if roi_button.clicked() {
                *roi_mode = !*roi_mode;
            }
            roi_button.on_hover_text(
                "Click to toggle ROI selection mode.\n\
                 When ON: drag on image draws ROI.\n\
                 Shortcut: Shift+drag always draws ROI.\n\
                 Drag inside existing ROI to move it."
            );

            if let Some(r) = roi {
                ui.separator();
                // Show cursor-in-roi indicator with theme-aware colors
                let roi_text = if cursor_in_roi {
                    format!("ROI: ({},{})→({},{}) [drag to move]", r.min_x, r.min_y, r.max_x, r.max_y)
                } else {
                    format!("ROI: ({},{})→({},{})", r.min_x, r.min_y, r.max_x, r.max_y)
                };
                let roi_color = colors::roi_label(ui, cursor_in_roi);
                ui.label(
                    egui::RichText::new(roi_text)
                        .small()
                        .color(roi_color),
                );
                if ui.small_button("Clear")
                    .on_hover_text("Remove ROI selection")
                    .clicked()
                {
                    *clear_roi_clicked = true;
                }
            }
        });
        height_used += 20.0;

        if !self.show {
            return height_used;
        }

        // Update cached histograms if slice changed or ROI changed
        let roi_changed = self.cached_roi != roi.copied();
        let needs_update = self.cached_slice_index != Some(slice_index) || roi_changed;

        if needs_update {
            if let Some(orig) = original_slice {
                self.cached_original = Some(HistogramData::compute_with_roi(orig, 256, roi));
            }
            if let Some(proc) = processed_slice {
                self.cached_processed = Some(HistogramData::compute_with_roi(proc, 256, roi));
            }
            // Compute difference histogram
            if let (Some(orig), Some(proc)) = (original_slice, processed_slice) {
                let diff: Array2<f32> = orig - proc;
                self.cached_difference = Some(HistogramData::compute_with_roi(&diff, 256, roi));
            }
            self.cached_slice_index = Some(slice_index);
            self.cached_roi = roi.copied();
        }

        let plot_height = 120.0; // Increased to accommodate legend
        height_used += plot_height + 45.0; // Plot + stats

        // Determine if we need twin axis display
        let show_main = self.show_original || self.show_processed;
        let show_diff = self.show_difference && self.cached_difference.is_some();

        if show_main && show_diff {
            // Stacked layout: main histogram on top, difference below
            ui.vertical(|ui| {
                ui.label(egui::RichText::new("Intensity Distribution").small());
                self.show_main_histogram(ui, plot_height * 0.8);

                ui.add_space(5.0);

                ui.label(egui::RichText::new("Difference Distribution").small());
                self.show_difference_histogram(ui, plot_height * 0.8);
            });
        } else if show_main {
            self.show_main_histogram(ui, plot_height);
        } else if show_diff {
            self.show_difference_histogram(ui, plot_height);
        }

        // Statistics
        self.show_statistics(ui);

        height_used
    }

    fn show_main_histogram(&self, ui: &mut egui::Ui, height: f32) {
        let mut charts = Vec::new();

        // Get theme-aware colors
        let original_color = colors::original_bar(ui, 150);
        let processed_color = colors::processed_bar(ui, 150);

        if self.show_original {
            if let Some(hist) = &self.cached_original {
                let bars: Vec<Bar> = hist
                    .counts
                    .iter()
                    .enumerate()
                    .map(|(i, &count)| {
                        let bin_width = (hist.max_val - hist.min_val) / hist.counts.len() as f32;
                        let x = hist.min_val + (i as f32 + 0.5) * bin_width;
                        Bar::new(x as f64, count as f64)
                            .width(bin_width as f64 * 0.8)
                    })
                    .collect();

                charts.push(
                    BarChart::new(bars)
                        .color(original_color)
                        .name("Original"),
                );
            }
        }

        if self.show_processed {
            if let Some(hist) = &self.cached_processed {
                let bars: Vec<Bar> = hist
                    .counts
                    .iter()
                    .enumerate()
                    .map(|(i, &count)| {
                        let bin_width = (hist.max_val - hist.min_val) / hist.counts.len() as f32;
                        let x = hist.min_val + (i as f32 + 0.5) * bin_width;
                        Bar::new(x as f64, count as f64)
                            .width(bin_width as f64 * 0.8)
                    })
                    .collect();

                charts.push(
                    BarChart::new(bars)
                        .color(processed_color)
                        .name("Processed"),
                );
            }
        }

        Plot::new("compare_main_histogram")
            .height(height)
            .show_axes([true, true])
            .show_grid(true)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .allow_boxed_zoom(false)
            .legend(Legend::default().position(Corner::LeftTop))
            .show(ui, |plot_ui| {
                for chart in charts {
                    plot_ui.bar_chart(chart);
                }
            });
    }

    fn show_difference_histogram(&self, ui: &mut egui::Ui, height: f32) {
        if let Some(hist) = &self.cached_difference {
            // Get theme-aware color
            let diff_color = colors::difference_bar(ui, 150);

            let bars: Vec<Bar> = hist
                .counts
                .iter()
                .enumerate()
                .map(|(i, &count)| {
                    let bin_width = (hist.max_val - hist.min_val) / hist.counts.len() as f32;
                    let x = hist.min_val + (i as f32 + 0.5) * bin_width;
                    Bar::new(x as f64, count as f64)
                        .width(bin_width as f64 * 0.8)
                })
                .collect();

            let chart = BarChart::new(bars)
                .color(diff_color)
                .name("Difference");

            Plot::new("compare_diff_histogram")
                .height(height)
                .show_axes([true, true])
                .show_grid(true)
                .allow_drag(false)
                .allow_zoom(false)
                .allow_scroll(false)
                .allow_boxed_zoom(false)
                .legend(Legend::default().position(Corner::LeftTop))
                .show(ui, |plot_ui| {
                    plot_ui.bar_chart(chart);
                });
        }
    }

    fn show_statistics(&self, ui: &mut egui::Ui) {
        // Get theme-aware colors
        let original_color = colors::original(ui);
        let processed_color = colors::processed(ui);
        let diff_color = colors::difference(ui);

        ui.horizontal(|ui| {
            if self.show_original {
                if let Some(hist) = &self.cached_original {
                    ui.label(
                        egui::RichText::new(format!(
                            "Orig: μ={:.3} σ={:.3}",
                            hist.stats.mean, hist.stats.std
                        ))
                        .small()
                        .color(original_color),
                    );
                }
            }

            if self.show_processed {
                if let Some(hist) = &self.cached_processed {
                    ui.label(
                        egui::RichText::new(format!(
                            "Proc: μ={:.3} σ={:.3}",
                            hist.stats.mean, hist.stats.std
                        ))
                        .small()
                        .color(processed_color),
                    );
                }
            }

            if self.show_difference {
                if let Some(hist) = &self.cached_difference {
                    ui.label(
                        egui::RichText::new(format!(
                            "Diff: μ={:.3} σ={:.3}",
                            hist.stats.mean, hist.stats.std
                        ))
                        .small()
                        .color(diff_color),
                    );
                }
            }
        });
    }
}
