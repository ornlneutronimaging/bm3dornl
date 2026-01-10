use crate::data::Volume3D;
use crate::ui::colormap::{Colormap, DivergingColormap};
use crate::ui::histogram::colors;
use crate::ui::slice_view::{ImageRoi, RoiState};
use crate::ui::window_level::WindowLevel;
use eframe::egui;
use ndarray::Array2;

/// Cursor information for comparison view display.
#[derive(Clone, Default)]
pub struct CompareCursorInfo {
    /// Image X coordinate (column)
    pub x: Option<usize>,
    /// Image Y coordinate (row)
    pub y: Option<usize>,
    /// Intensity value from original image
    pub orig_intensity: Option<f32>,
    /// Intensity value from processed image
    pub proc_intensity: Option<f32>,
    /// Difference value (original - processed)
    pub diff_intensity: Option<f32>,
}

impl CompareCursorInfo {
    pub fn clear(&mut self) {
        self.x = None;
        self.y = None;
        self.orig_intensity = None;
        self.proc_intensity = None;
        self.diff_intensity = None;
    }

    pub fn is_valid(&self) -> bool {
        self.x.is_some() && self.y.is_some()
    }
}

/// Comparison view showing Original | Processed | Difference side by side
/// with horizontal colorbars below
pub struct CompareView {
    // Shared slice index for all panels
    current_slice: usize,

    // Textures for each panel
    original_texture: Option<egui::TextureHandle>,
    processed_texture: Option<egui::TextureHandle>,
    difference_texture: Option<egui::TextureHandle>,

    // Cache keys
    cached_slice_index: Option<usize>,
    cached_colormap: Option<Colormap>,
    cached_window_level: Option<(f32, f32)>,

    // Difference range (symmetric around zero)
    diff_max: f32,

    // Cursor tracking
    cursor_info: CompareCursorInfo,

    // ROI selection (shared across all three panels)
    roi_state: RoiState,
    /// ROI drawing mode active (drag draws ROI without Shift).
    pub roi_mode: bool,
    /// Is cursor currently inside the ROI? (for visual feedback)
    cursor_in_roi: bool,

    // For drag detection (pan vs ROI)
    is_dragging: bool,
    last_drag_pos: Option<egui::Pos2>,

    // Cached image rect for coordinate conversion
    cached_image_rect: Option<egui::Rect>,
    cached_image_dims: Option<(usize, usize)>,
}

impl Default for CompareView {
    fn default() -> Self {
        Self::new()
    }
}

impl CompareView {
    pub fn new() -> Self {
        Self {
            current_slice: 0,
            original_texture: None,
            processed_texture: None,
            difference_texture: None,
            cached_slice_index: None,
            cached_colormap: None,
            cached_window_level: None,
            diff_max: 1.0,
            cursor_info: CompareCursorInfo::default(),
            roi_state: RoiState::new(),
            roi_mode: false,
            cursor_in_roi: false,
            is_dragging: false,
            last_drag_pos: None,
            cached_image_rect: None,
            cached_image_dims: None,
        }
    }

    pub fn current_slice(&self) -> usize {
        self.current_slice
    }

    pub fn reset(&mut self) {
        self.current_slice = 0;
        self.original_texture = None;
        self.processed_texture = None;
        self.difference_texture = None;
        self.cached_slice_index = None;
        self.cached_colormap = None;
        self.cached_window_level = None;
        self.diff_max = 1.0;
        self.cursor_info.clear();
        self.roi_state.clear();
        self.roi_mode = false;
        self.cursor_in_roi = false;
        self.is_dragging = false;
        self.last_drag_pos = None;
        self.cached_image_rect = None;
        self.cached_image_dims = None;
    }

    /// Check if cursor is currently inside the ROI (for UI feedback).
    pub fn is_cursor_in_roi(&self) -> bool {
        self.cursor_in_roi
    }

    /// Get the current ROI, if any.
    pub fn roi(&self) -> Option<&ImageRoi> {
        self.roi_state.roi.as_ref()
    }

    /// Clear the current ROI.
    pub fn clear_roi(&mut self) {
        self.roi_state.clear();
    }

    /// Show comparison view. Returns true if slice changed.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        original: &Volume3D,
        processed: &Volume3D,
        colormap: Colormap,
        window_level: &WindowLevel,
        keep_aspect_ratio: bool,
    ) -> bool {
        self.show_with_slice_data(
            ui,
            original,
            processed,
            None,
            None,
            colormap,
            window_level,
            keep_aspect_ratio,
        )
    }

    /// Show comparison view with optional slice data for cursor tracking.
    /// Returns true if slice changed.
    pub fn show_with_slice_data(
        &mut self,
        ui: &mut egui::Ui,
        original: &Volume3D,
        processed: &Volume3D,
        orig_slice: Option<&Array2<f32>>,
        proc_slice: Option<&Array2<f32>>,
        colormap: Colormap,
        window_level: &WindowLevel,
        keep_aspect_ratio: bool,
    ) -> bool {
        let num_slices = original.num_slices();
        let mut slice_changed = false;

        // Clamp current slice
        if self.current_slice >= num_slices {
            self.current_slice = num_slices.saturating_sub(1);
        }

        // Navigation controls (shared for all panels)
        ui.horizontal(|ui| {
            if ui.button("◄").on_hover_text("Previous slice").clicked() && self.current_slice > 0
            {
                self.current_slice -= 1;
                slice_changed = true;
            }

            let slider_response = ui
                .add(
                    egui::Slider::new(&mut self.current_slice, 0..=num_slices.saturating_sub(1))
                        .text("")
                        .show_value(false),
                )
                .on_hover_text("Navigate through slices along the selected axis");
            if slider_response.changed() {
                slice_changed = true;
            }

            if ui.button("►").on_hover_text("Next slice").clicked()
                && self.current_slice < num_slices - 1
            {
                self.current_slice += 1;
                slice_changed = true;
            }

            let axis_mapping = original.axis_mapping();
            ui.label(format!(
                "Slice D{}: {} / {}",
                axis_mapping.slice_axis,
                self.current_slice + 1,
                num_slices
            ));
        });

        // Check if we need to update textures
        let wl_key = (window_level.min, window_level.max);
        let needs_update = self.cached_slice_index != Some(self.current_slice)
            || self.cached_colormap != Some(colormap)
            || self.cached_window_level != Some(wl_key)
            || self.original_texture.is_none();

        if needs_update {
            self.update_textures(ui.ctx(), original, processed, colormap, window_level);
        }

        // Get image dimensions for aspect ratio calculation
        let (img_width, img_height) = original.display_dimensions();
        let img_aspect = img_width as f32 / img_height as f32;

        // Layout constants
        let available_size = ui.available_size();
        let colorbar_height = 30.0;
        let label_height = 20.0;
        let status_bar_height = 20.0;
        let spacing = 5.0;

        // Calculate panel dimensions
        let panel_width = (available_size.x - 2.0 * spacing) / 3.0;
        let image_height =
            available_size.y - colorbar_height - label_height - status_bar_height - 3.0 * spacing;

        // Row 1: Labels
        ui.horizontal(|ui| {
            ui.allocate_ui(egui::vec2(panel_width, label_height), |ui| {
                ui.label(egui::RichText::new("Original").strong());
            });
            ui.add_space(spacing);
            ui.allocate_ui(egui::vec2(panel_width, label_height), |ui| {
                ui.label(egui::RichText::new("Processed").strong());
            });
            ui.add_space(spacing);
            ui.allocate_ui(egui::vec2(panel_width, label_height), |ui| {
                ui.label(egui::RichText::new("Difference").strong());
            });
        });

        // Row 2: Images with cursor tracking and ROI interaction
        // Clear cursor info before checking panels
        self.cursor_info.clear();

        // We need to collect panel info first, then handle interactions outside the closure
        // because handle_mouse_input needs &mut self
        let mut panel_data: Vec<(egui::Response, Option<egui::Rect>, egui::Painter)> = Vec::new();

        ui.horizontal(|ui| {
            // Original image
            let orig_result = self.draw_image_panel_with_response(
                ui,
                &self.original_texture,
                panel_width,
                image_height,
                img_aspect,
                keep_aspect_ratio,
            );
            panel_data.push(orig_result);
            ui.add_space(spacing);

            // Processed image
            let proc_result = self.draw_image_panel_with_response(
                ui,
                &self.processed_texture,
                panel_width,
                image_height,
                img_aspect,
                keep_aspect_ratio,
            );
            panel_data.push(proc_result);
            ui.add_space(spacing);

            // Difference image
            let diff_result = self.draw_image_panel_with_response(
                ui,
                &self.difference_texture,
                panel_width,
                image_height,
                img_aspect,
                keep_aspect_ratio,
            );
            panel_data.push(diff_result);
        });

        // Cache image dimensions for ROI coordinate conversion
        self.cached_image_dims = Some((img_width, img_height));

        // Find which panel is being interacted with (if any)
        // Use the first panel that has valid image rect for ROI interactions
        let mut active_response: Option<&egui::Response> = None;
        let mut active_image_rect: Option<egui::Rect> = None;

        for (response, image_rect, _) in &panel_data {
            if let Some(rect) = image_rect {
                // Check if this panel is being hovered or dragged
                if response.hovered() || response.dragged() {
                    active_response = Some(response);
                    active_image_rect = Some(*rect);
                    self.cached_image_rect = Some(*rect);
                    break;
                }
            }
        }

        // Handle ROI mouse interactions on the active panel
        if let (Some(response), Some(image_rect)) = (active_response, active_image_rect) {
            // Clone the response to avoid borrow issues
            let response_clone = response.clone();
            self.handle_mouse_input(&response_clone, image_rect, img_width, img_height);
        } else {
            // Not hovering any panel - reset cursor_in_roi and handle drag end
            self.cursor_in_roi = false;
            if self.roi_state.drawing {
                self.roi_state.finish_draw();
            }
            if self.roi_state.moving {
                self.roi_state.finish_move();
            }
        }

        // Track cursor position and look up intensities
        for (response, image_rect, _) in &panel_data {
            if let Some((x, y)) = self.get_image_coords(response, image_rect, img_width, img_height)
            {
                self.cursor_info.x = Some(x);
                self.cursor_info.y = Some(y);

                // Look up intensities
                if let Some(orig_data) = orig_slice {
                    if y < orig_data.nrows() && x < orig_data.ncols() {
                        self.cursor_info.orig_intensity = Some(orig_data[[y, x]]);
                    }
                }
                if let Some(proc_data) = proc_slice {
                    if y < proc_data.nrows() && x < proc_data.ncols() {
                        self.cursor_info.proc_intensity = Some(proc_data[[y, x]]);
                    }
                }
                // Compute difference
                if let (Some(orig_val), Some(proc_val)) = (
                    self.cursor_info.orig_intensity,
                    self.cursor_info.proc_intensity,
                ) {
                    self.cursor_info.diff_intensity = Some(orig_val - proc_val);
                }
                break; // Only process the first hovered panel
            }
        }

        // Draw ROI overlay on all three panels
        for (_, image_rect, painter) in &panel_data {
            if let Some(rect) = image_rect {
                self.draw_roi_overlay(painter, *rect, img_width, img_height);
            }
        }

        // Row 3: Cursor status bar
        self.draw_cursor_status_bar(ui, available_size.x, status_bar_height);

        // Row 4: Colorbars
        ui.horizontal(|ui| {
            // Shared colorbar for Original and Processed (spans 2 panels + spacing)
            let shared_colorbar_width = panel_width * 2.0 + spacing;
            self.draw_horizontal_colorbar(
                ui,
                shared_colorbar_width,
                colorbar_height,
                colormap,
                window_level,
            );
            ui.add_space(spacing);

            // Difference colorbar (RdBu)
            self.draw_horizontal_diff_colorbar(ui, panel_width, colorbar_height);
        });

        slice_changed
    }

    /// Get image coordinates from hover position.
    fn get_image_coords(
        &self,
        response: &egui::Response,
        image_rect: &Option<egui::Rect>,
        img_width: usize,
        img_height: usize,
    ) -> Option<(usize, usize)> {
        let hover_pos = response.hover_pos()?;
        let rect = (*image_rect)?;

        if !rect.contains(hover_pos) {
            return None;
        }

        // Convert screen position to normalized image coordinates [0, 1]
        let norm_x = (hover_pos.x - rect.left()) / rect.width();
        let norm_y = (hover_pos.y - rect.top()) / rect.height();

        // Convert to pixel coordinates
        let pixel_x = (norm_x * img_width as f32).floor() as i32;
        let pixel_y = (norm_y * img_height as f32).floor() as i32;

        // Bounds check
        if pixel_x >= 0 && pixel_x < img_width as i32 && pixel_y >= 0 && pixel_y < img_height as i32
        {
            Some((pixel_x as usize, pixel_y as usize))
        } else {
            None
        }
    }

    /// Draw cursor status bar showing coordinates and intensities.
    fn draw_cursor_status_bar(&self, ui: &mut egui::Ui, width: f32, height: f32) {
        // Get theme-aware colors
        let original_color = colors::original(ui);
        let processed_color = colors::processed(ui);
        let diff_color = colors::difference(ui);

        ui.allocate_ui(egui::vec2(width, height), |ui| {
            ui.horizontal(|ui| {
                if self.cursor_info.is_valid() {
                    let x = self.cursor_info.x.unwrap();
                    let y = self.cursor_info.y.unwrap();

                    ui.label(
                        egui::RichText::new(format!("(x: {}, y: {})", x, y))
                            .monospace()
                            .small(),
                    );

                    ui.separator();

                    // Original intensity (blue to match histogram)
                    if let Some(val) = self.cursor_info.orig_intensity {
                        ui.label(
                            egui::RichText::new(format!("Orig: {:.4}", val))
                                .monospace()
                                .small()
                                .color(original_color),
                        );
                    } else {
                        ui.label(egui::RichText::new("Orig: ---").monospace().small().weak());
                    }

                    ui.separator();

                    // Processed intensity (orange to match histogram)
                    if let Some(val) = self.cursor_info.proc_intensity {
                        ui.label(
                            egui::RichText::new(format!("Proc: {:.4}", val))
                                .monospace()
                                .small()
                                .color(processed_color),
                        );
                    } else {
                        ui.label(egui::RichText::new("Proc: ---").monospace().small().weak());
                    }

                    ui.separator();

                    // Difference (red to match histogram)
                    if let Some(val) = self.cursor_info.diff_intensity {
                        ui.label(
                            egui::RichText::new(format!("Diff: {:.4}", val))
                                .monospace()
                                .small()
                                .color(diff_color),
                        );
                    } else {
                        ui.label(egui::RichText::new("Diff: ---").monospace().small().weak());
                    }
                } else {
                    ui.label(
                        egui::RichText::new("(x: ---, y: ---)  Orig: ---  Proc: ---  Diff: ---")
                            .monospace()
                            .small()
                            .weak(),
                    );
                }
            });
        });
    }

    /// Draw image panel and return response + computed image rect + painter for ROI drawing.
    fn draw_image_panel_with_response(
        &self,
        ui: &mut egui::Ui,
        texture: &Option<egui::TextureHandle>,
        width: f32,
        height: f32,
        img_aspect: f32,
        keep_aspect_ratio: bool,
    ) -> (egui::Response, Option<egui::Rect>, egui::Painter) {
        let (response, painter) =
            ui.allocate_painter(egui::vec2(width, height), egui::Sense::click_and_drag());

        let mut computed_image_rect = None;

        if let Some(tex) = texture {
            let panel_rect = response.rect;

            // Calculate image rect based on aspect ratio setting
            let image_rect = if keep_aspect_ratio {
                let panel_aspect = panel_rect.width() / panel_rect.height();
                let (img_w, img_h) = if img_aspect > panel_aspect {
                    (panel_rect.width(), panel_rect.width() / img_aspect)
                } else {
                    (panel_rect.height() * img_aspect, panel_rect.height())
                };
                egui::Rect::from_center_size(panel_rect.center(), egui::vec2(img_w, img_h))
            } else {
                panel_rect
            };

            computed_image_rect = Some(image_rect);

            painter.image(
                tex.id(),
                image_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }

        // Border
        painter.rect_stroke(
            response.rect,
            0.0,
            egui::Stroke::new(1.0, egui::Color32::GRAY),
        );

        (response, computed_image_rect, painter)
    }

    fn draw_horizontal_colorbar(
        &self,
        ui: &mut egui::Ui,
        width: f32,
        height: f32,
        colormap: Colormap,
        window_level: &WindowLevel,
    ) {
        ui.vertical(|ui| {
            // Colorbar gradient
            let (response, painter) =
                ui.allocate_painter(egui::vec2(width, height - 15.0), egui::Sense::hover());

            let rect = response.rect;
            let lut = colormap.generate_lut();
            let num_steps = 256;
            let step_width = rect.width() / num_steps as f32;

            for (i, [r, g, b]) in lut.iter().enumerate().take(num_steps) {
                let color = egui::Color32::from_rgb(*r, *g, *b);

                let x_start = rect.left() + i as f32 * step_width;
                let step_rect = egui::Rect::from_min_size(
                    egui::pos2(x_start, rect.top()),
                    egui::vec2(step_width + 1.0, rect.height()),
                );
                painter.rect_filled(step_rect, 0.0, color);
            }

            // Border
            painter.rect_stroke(rect, 0.0, egui::Stroke::new(1.0, egui::Color32::GRAY));

            // Labels row
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(format!("{:.2}", window_level.min)).small());
                ui.add_space(width - 80.0);
                ui.label(egui::RichText::new(format!("{:.2}", window_level.max)).small());
            });
        });
    }

    fn draw_horizontal_diff_colorbar(&self, ui: &mut egui::Ui, width: f32, height: f32) {
        ui.vertical(|ui| {
            // Colorbar gradient (RdBu: blue -> white -> red)
            let (response, painter) =
                ui.allocate_painter(egui::vec2(width, height - 15.0), egui::Sense::hover());

            let rect = response.rect;
            let num_steps = 256;
            let step_width = rect.width() / num_steps as f32;

            for i in 0..num_steps {
                // Map [0, 255] to [-1, 1] for RdBu colormap
                let t = (i as f32 / 255.0) * 2.0 - 1.0;
                let [r, g, b] = DivergingColormap::rdbu(t);
                let color = egui::Color32::from_rgb(r, g, b);

                let x_start = rect.left() + i as f32 * step_width;
                let step_rect = egui::Rect::from_min_size(
                    egui::pos2(x_start, rect.top()),
                    egui::vec2(step_width + 1.0, rect.height()),
                );
                painter.rect_filled(step_rect, 0.0, color);
            }

            // Border
            painter.rect_stroke(rect, 0.0, egui::Stroke::new(1.0, egui::Color32::GRAY));

            // Labels row: -max | 0 | +max
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(format!("{:.2}", -self.diff_max)).small());
                ui.add_space((width - 60.0) / 2.0 - 10.0);
                ui.label(egui::RichText::new("0").small());
                ui.add_space((width - 60.0) / 2.0 - 10.0);
                ui.label(egui::RichText::new(format!("+{:.2}", self.diff_max)).small());
            });
        });
    }

    fn update_textures(
        &mut self,
        ctx: &egui::Context,
        original: &Volume3D,
        processed: &Volume3D,
        colormap: Colormap,
        window_level: &WindowLevel,
    ) {
        let orig_slice = original.get_slice(self.current_slice);
        let proc_slice = processed.get_slice(self.current_slice);

        if let (Some(orig), Some(proc)) = (orig_slice, proc_slice) {
            let (height, width) = (orig.nrows(), orig.ncols());
            let lut = colormap.generate_lut();

            // Original texture
            let orig_rgba: Vec<u8> = orig
                .iter()
                .flat_map(|&val| {
                    let normalized = window_level.normalize(val);
                    let lut_index = (normalized * 255.0) as usize;
                    let [r, g, b] = lut[lut_index.min(255)];
                    [r, g, b, 255]
                })
                .collect();

            let orig_image = egui::ColorImage::from_rgba_unmultiplied([width, height], &orig_rgba);
            self.original_texture = Some(ctx.load_texture(
                "compare_original",
                orig_image,
                egui::TextureOptions::default(),
            ));

            // Processed texture
            let proc_rgba: Vec<u8> = proc
                .iter()
                .flat_map(|&val| {
                    let normalized = window_level.normalize(val);
                    let lut_index = (normalized * 255.0) as usize;
                    let [r, g, b] = lut[lut_index.min(255)];
                    [r, g, b, 255]
                })
                .collect();

            let proc_image = egui::ColorImage::from_rgba_unmultiplied([width, height], &proc_rgba);
            self.processed_texture = Some(ctx.load_texture(
                "compare_processed",
                proc_image,
                egui::TextureOptions::default(),
            ));

            // Difference: Original - Processed
            let diff: Array2<f32> = &orig - &proc;

            // Find max absolute value for symmetric range
            let mut max_abs = 0.0f32;
            for &val in diff.iter() {
                if val.is_finite() {
                    max_abs = max_abs.max(val.abs());
                }
            }
            self.diff_max = if max_abs > 0.0 { max_abs } else { 1.0 };

            // Difference texture with RdBu colormap
            let rdbu_lut = DivergingColormap::generate_lut();
            let diff_rgba: Vec<u8> = diff
                .iter()
                .flat_map(|&val| {
                    // Normalize to [-1, 1] based on symmetric range
                    let normalized = if self.diff_max > 0.0 {
                        val / self.diff_max
                    } else {
                        0.0
                    };
                    // Map [-1, 1] to [0, 255] for LUT lookup
                    let lut_index = ((normalized + 1.0) * 0.5 * 255.0) as usize;
                    let [r, g, b] = rdbu_lut[lut_index.clamp(0, 255)];
                    [r, g, b, 255]
                })
                .collect();

            let diff_image = egui::ColorImage::from_rgba_unmultiplied([width, height], &diff_rgba);
            self.difference_texture = Some(ctx.load_texture(
                "compare_difference",
                diff_image,
                egui::TextureOptions::default(),
            ));

            // Update cache
            self.cached_slice_index = Some(self.current_slice);
            self.cached_colormap = Some(colormap);
            self.cached_window_level = Some((window_level.min, window_level.max));
        }
    }

    /// Convert screen position to image pixel coordinates.
    fn screen_to_image(
        &self,
        screen_pos: egui::Pos2,
        image_rect: egui::Rect,
        img_width: usize,
        img_height: usize,
    ) -> Option<(usize, usize)> {
        if !image_rect.contains(screen_pos) {
            return None;
        }

        // Convert screen position to normalized image coordinates [0, 1]
        let norm_x = (screen_pos.x - image_rect.left()) / image_rect.width();
        let norm_y = (screen_pos.y - image_rect.top()) / image_rect.height();

        // Convert to pixel coordinates
        let pixel_x = (norm_x * img_width as f32).floor() as i32;
        let pixel_y = (norm_y * img_height as f32).floor() as i32;

        // Bounds check
        if pixel_x >= 0 && pixel_x < img_width as i32 && pixel_y >= 0 && pixel_y < img_height as i32
        {
            Some((pixel_x as usize, pixel_y as usize))
        } else {
            None
        }
    }

    /// Convert image pixel coordinates to screen position.
    fn image_to_screen(
        &self,
        img_x: usize,
        img_y: usize,
        image_rect: egui::Rect,
        img_width: usize,
        img_height: usize,
    ) -> egui::Pos2 {
        let norm_x = img_x as f32 / img_width as f32;
        let norm_y = img_y as f32 / img_height as f32;

        egui::pos2(
            image_rect.left() + norm_x * image_rect.width(),
            image_rect.top() + norm_y * image_rect.height(),
        )
    }

    /// Handle mouse input for ROI drawing/moving across all panels.
    fn handle_mouse_input(
        &mut self,
        response: &egui::Response,
        image_rect: egui::Rect,
        img_width: usize,
        img_height: usize,
    ) {
        // Check if Shift is held for ROI drawing mode
        let shift_held = response.ctx.input(|i| i.modifiers.shift);

        // Determine if we should draw ROI: either roi_mode is ON or Shift is held
        let should_draw_roi = self.roi_mode || shift_held;

        // Update cursor_in_roi for visual feedback
        if response.hovered() {
            if let Some(pointer_pos) = response.hover_pos() {
                if let Some((img_x, img_y)) =
                    self.screen_to_image(pointer_pos, image_rect, img_width, img_height)
                {
                    self.cursor_in_roi = self.roi_state.cursor_inside_roi(img_x, img_y);
                } else {
                    self.cursor_in_roi = false;
                }
            } else {
                self.cursor_in_roi = false;
            }
        } else {
            self.cursor_in_roi = false;
        }

        // Handle drag
        if response.dragged() {
            if let Some(pointer_pos) = response.interact_pointer_pos() {
                if let Some((img_x, img_y)) =
                    self.screen_to_image(pointer_pos, image_rect, img_width, img_height)
                {
                    // Check if we're already in a dragging operation
                    if self.roi_state.drawing {
                        // Continue drawing ROI
                        self.roi_state.update_draw(img_x, img_y);
                        self.is_dragging = false;
                        self.last_drag_pos = None;
                    } else if self.roi_state.moving {
                        // Continue moving ROI
                        self.roi_state
                            .update_move(img_x, img_y, img_width, img_height);
                        self.is_dragging = false;
                        self.last_drag_pos = None;
                    } else {
                        // Starting a new drag - determine what to do
                        let inside_roi = self.roi_state.cursor_inside_roi(img_x, img_y);

                        if inside_roi {
                            // Drag inside existing ROI → always move ROI (even in ROI mode)
                            self.roi_state.start_move(img_x, img_y);
                            self.is_dragging = false;
                            self.last_drag_pos = None;
                        } else if should_draw_roi {
                            // ROI drawing mode (roi_mode ON or Shift held) outside ROI → draw new ROI
                            self.roi_state.start_draw(img_x, img_y);
                            self.is_dragging = false;
                            self.last_drag_pos = None;
                        }
                        // Note: No panning in compare view, just ignore non-ROI drags
                    }
                }
            }
        } else {
            // Drag ended
            if self.roi_state.drawing {
                self.roi_state.finish_draw();
            }
            if self.roi_state.moving {
                self.roi_state.finish_move();
            }
            self.is_dragging = false;
            self.last_drag_pos = None;
        }
    }

    /// Draw ROI overlay on a single image panel.
    fn draw_roi_overlay(
        &self,
        painter: &egui::Painter,
        image_rect: egui::Rect,
        img_width: usize,
        img_height: usize,
    ) {
        // Get ROI to draw (either finalized or currently drawing)
        let roi_to_draw = self.roi_state.drawing_rect().or(self.roi_state.roi);

        if let Some(roi) = roi_to_draw {
            // Convert image coordinates to screen coordinates
            let top_left =
                self.image_to_screen(roi.min_x, roi.min_y, image_rect, img_width, img_height);
            let bottom_right = self.image_to_screen(
                roi.max_x + 1, // +1 because max is inclusive
                roi.max_y + 1,
                image_rect,
                img_width,
                img_height,
            );

            let screen_rect = egui::Rect::from_min_max(top_left, bottom_right);

            // Draw semi-transparent fill - brighter when cursor is inside (hover feedback)
            let fill_alpha = if self.cursor_in_roi && !self.roi_state.drawing {
                60
            } else {
                30
            };
            let fill_color = egui::Color32::from_rgba_unmultiplied(255, 255, 0, fill_alpha);
            painter.rect_filled(screen_rect, 0.0, fill_color);

            // Draw border - different colors for different states
            let stroke_color = if self.roi_state.drawing {
                egui::Color32::from_rgb(0, 255, 255) // Cyan while drawing
            } else if self.roi_state.moving {
                egui::Color32::from_rgb(0, 255, 0) // Green while moving
            } else if self.cursor_in_roi {
                egui::Color32::from_rgb(255, 165, 0) // Orange when hovering inside (movable)
            } else {
                egui::Color32::from_rgb(255, 255, 0) // Yellow when finalized
            };
            let stroke_width = if self.cursor_in_roi || self.roi_state.moving {
                3.0
            } else {
                2.0
            };
            painter.rect_stroke(
                screen_rect,
                0.0,
                egui::Stroke::new(stroke_width, stroke_color),
            );
        }
    }
}
