use crate::data::{AxisMapping, Volume3D};
use crate::ui::colormap::Colormap;
use crate::ui::window_level::WindowLevel;
use eframe::egui;
use ndarray::Array2;

/// Region of Interest in image coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImageRoi {
    /// Minimum X coordinate (column)
    pub min_x: usize,
    /// Minimum Y coordinate (row)
    pub min_y: usize,
    /// Maximum X coordinate (column, inclusive)
    pub max_x: usize,
    /// Maximum Y coordinate (row, inclusive)
    pub max_y: usize,
}

impl ImageRoi {
    /// Create a new ROI from two corner points, normalizing to ensure min <= max.
    pub fn from_corners(x1: usize, y1: usize, x2: usize, y2: usize) -> Self {
        Self {
            min_x: x1.min(x2),
            min_y: y1.min(y2),
            max_x: x1.max(x2),
            max_y: y1.max(y2),
        }
    }

    /// Get the width of the ROI in pixels.
    pub fn width(&self) -> usize {
        self.max_x.saturating_sub(self.min_x) + 1
    }

    /// Get the height of the ROI in pixels.
    pub fn height(&self) -> usize {
        self.max_y.saturating_sub(self.min_y) + 1
    }

    /// Get the total number of pixels in the ROI.
    pub fn pixel_count(&self) -> usize {
        self.width() * self.height()
    }

    /// Check if the ROI is valid (non-zero area).
    pub fn is_valid(&self) -> bool {
        self.width() > 1 && self.height() > 1
    }
}

/// State for ROI selection.
#[derive(Clone, Default)]
pub struct RoiState {
    /// Current ROI in image coordinates (None = no ROI).
    pub roi: Option<ImageRoi>,
    /// Is user currently drawing an ROI?
    pub drawing: bool,
    /// Start point of current draw operation (image coords).
    draw_start: Option<(usize, usize)>,
    /// Current drag end point during drawing (image coords).
    draw_end: Option<(usize, usize)>,
    /// Is user currently moving an existing ROI?
    pub moving: bool,
    /// Offset from cursor to ROI top-left when move started.
    move_offset: Option<(i32, i32)>,
}

impl RoiState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.roi = None;
        self.drawing = false;
        self.draw_start = None;
        self.draw_end = None;
        self.moving = false;
        self.move_offset = None;
    }

    /// Start drawing a new ROI at the given image coordinates.
    pub fn start_draw(&mut self, x: usize, y: usize) {
        self.drawing = true;
        self.draw_start = Some((x, y));
        self.draw_end = Some((x, y));
        self.roi = None; // Clear existing ROI when starting new draw
    }

    /// Update the draw end point.
    pub fn update_draw(&mut self, x: usize, y: usize) {
        if self.drawing {
            self.draw_end = Some((x, y));
        }
    }

    /// Finish drawing and create the ROI if valid.
    pub fn finish_draw(&mut self) {
        if self.drawing {
            if let (Some((x1, y1)), Some((x2, y2))) = (self.draw_start, self.draw_end) {
                let roi = ImageRoi::from_corners(x1, y1, x2, y2);
                if roi.is_valid() {
                    self.roi = Some(roi);
                }
            }
        }
        self.drawing = false;
        self.draw_start = None;
        self.draw_end = None;
    }

    /// Get the current drawing rectangle (for preview), if any.
    pub fn drawing_rect(&self) -> Option<ImageRoi> {
        if self.drawing {
            if let (Some((x1, y1)), Some((x2, y2))) = (self.draw_start, self.draw_end) {
                return Some(ImageRoi::from_corners(x1, y1, x2, y2));
            }
        }
        None
    }

    /// Check if a point is inside the current ROI.
    pub fn cursor_inside_roi(&self, x: usize, y: usize) -> bool {
        if let Some(roi) = &self.roi {
            x >= roi.min_x && x <= roi.max_x && y >= roi.min_y && y <= roi.max_y
        } else {
            false
        }
    }

    /// Start moving the ROI from the given cursor position.
    pub fn start_move(&mut self, cursor_x: usize, cursor_y: usize) {
        if let Some(roi) = &self.roi {
            self.moving = true;
            self.move_offset = Some((
                cursor_x as i32 - roi.min_x as i32,
                cursor_y as i32 - roi.min_y as i32,
            ));
        }
    }

    /// Update ROI position during move, clamping to image bounds.
    pub fn update_move(
        &mut self,
        cursor_x: usize,
        cursor_y: usize,
        img_width: usize,
        img_height: usize,
    ) {
        if self.moving {
            if let (Some(roi), Some((off_x, off_y))) = (&self.roi, self.move_offset) {
                let width = roi.width();
                let height = roi.height();

                // Calculate new top-left position
                let new_min_x = (cursor_x as i32 - off_x).max(0) as usize;
                let new_min_y = (cursor_y as i32 - off_y).max(0) as usize;

                // Clamp to image bounds
                let new_min_x = new_min_x.min(img_width.saturating_sub(width));
                let new_min_y = new_min_y.min(img_height.saturating_sub(height));

                self.roi = Some(ImageRoi {
                    min_x: new_min_x,
                    min_y: new_min_y,
                    max_x: new_min_x + width - 1,
                    max_y: new_min_y + height - 1,
                });
            }
        }
    }

    /// Finish moving the ROI.
    pub fn finish_move(&mut self) {
        self.moving = false;
        self.move_offset = None;
    }
}

/// Cursor information for display.
#[derive(Clone, Default)]
pub struct CursorInfo {
    /// Image X coordinate (column)
    pub x: Option<usize>,
    /// Image Y coordinate (row)
    pub y: Option<usize>,
    /// Intensity value at cursor position
    pub intensity: Option<f32>,
}

impl CursorInfo {
    pub fn clear(&mut self) {
        self.x = None;
        self.y = None;
        self.intensity = None;
    }

    pub fn is_valid(&self) -> bool {
        self.x.is_some() && self.y.is_some()
    }
}

/// Pan/zoom state for image display.
#[derive(Clone, Copy)]
pub struct ViewTransform {
    /// Zoom level (1.0 = 100%)
    pub zoom: f32,
    /// Pan offset in image coordinates
    pub pan: egui::Vec2,
}

impl Default for ViewTransform {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
        }
    }
}

impl ViewTransform {
    pub fn reset(&mut self) {
        self.zoom = 1.0;
        self.pan = egui::Vec2::ZERO;
    }

    pub fn zoom_percent(&self) -> i32 {
        (self.zoom * 100.0).round() as i32
    }
}

/// Slice viewer with navigation, pan/zoom, colormap, and window/level.
pub struct SliceViewer {
    current_slice: usize,
    texture_handle: Option<egui::TextureHandle>,
    cached_slice_index: Option<usize>,
    cached_axis_mapping: Option<AxisMapping>,
    cached_colormap: Option<Colormap>,
    cached_window_level: Option<(f32, f32)>,

    // View transform for pan/zoom
    view_transform: ViewTransform,
    // For drag detection
    is_dragging: bool,
    last_drag_pos: Option<egui::Pos2>,

    // Cursor tracking
    cursor_info: CursorInfo,
    // Cached image rect for cursor coordinate conversion
    cached_image_rect: Option<egui::Rect>,
    cached_image_dims: Option<(usize, usize)>,

    // ROI selection
    roi_state: RoiState,
    /// ROI drawing mode active (drag draws ROI without Shift).
    pub roi_mode: bool,
    /// Is cursor currently inside the ROI? (for visual feedback)
    cursor_in_roi: bool,
}

impl Default for SliceViewer {
    fn default() -> Self {
        Self::new()
    }
}

impl SliceViewer {
    pub fn new() -> Self {
        Self {
            current_slice: 0,
            texture_handle: None,
            cached_slice_index: None,
            cached_axis_mapping: None,
            cached_colormap: None,
            cached_window_level: None,
            view_transform: ViewTransform::default(),
            is_dragging: false,
            last_drag_pos: None,
            cursor_info: CursorInfo::default(),
            cached_image_rect: None,
            cached_image_dims: None,
            roi_state: RoiState::new(),
            roi_mode: false,
            cursor_in_roi: false,
        }
    }

    pub fn current_slice(&self) -> usize {
        self.current_slice
    }

    /// Reset viewer state (call when loading new data).
    pub fn reset(&mut self) {
        self.current_slice = 0;
        self.texture_handle = None;
        self.cached_slice_index = None;
        self.cached_axis_mapping = None;
        self.cached_colormap = None;
        self.cached_window_level = None;
        self.view_transform.reset();
        self.is_dragging = false;
        self.last_drag_pos = None;
        self.cursor_info.clear();
        self.cached_image_rect = None;
        self.cached_image_dims = None;
        self.roi_state.clear();
        self.roi_mode = false;
        self.cursor_in_roi = false;
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

    pub fn view_transform(&self) -> &ViewTransform {
        &self.view_transform
    }

    pub fn reset_view(&mut self) {
        self.view_transform.reset();
    }

    /// Show slice viewer with navigation. Returns true if slice changed.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        volume: &Volume3D,
        colormap: Colormap,
        window_level: &WindowLevel,
        keep_aspect_ratio: bool,
    ) -> bool {
        self.show_with_slice_data(ui, volume, None, colormap, window_level, keep_aspect_ratio)
    }

    /// Show slice viewer with navigation and optional slice data for cursor tracking.
    /// Returns true if slice changed.
    pub fn show_with_slice_data(
        &mut self,
        ui: &mut egui::Ui,
        volume: &Volume3D,
        slice_data: Option<&Array2<f32>>,
        colormap: Colormap,
        window_level: &WindowLevel,
        keep_aspect_ratio: bool,
    ) -> bool {
        let num_slices = volume.num_slices();
        let axis_mapping = volume.axis_mapping();
        let mut slice_changed = false;

        // Clamp current slice to valid range
        if self.current_slice >= num_slices {
            self.current_slice = num_slices.saturating_sub(1);
        }

        // Navigation controls
        ui.horizontal(|ui| {
            // Step backward button
            if ui.button("◄").on_hover_text("Previous slice").clicked() && self.current_slice > 0
            {
                self.current_slice -= 1;
                slice_changed = true;
            }

            // Slider
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

            // Step forward button
            if ui.button("►").on_hover_text("Next slice").clicked()
                && self.current_slice < num_slices - 1
            {
                self.current_slice += 1;
                slice_changed = true;
            }

            // Slice label
            ui.label(format!(
                "Slice D{}: {} / {}",
                axis_mapping.slice_axis,
                self.current_slice + 1,
                num_slices
            ));

            ui.separator();

            // Zoom indicator and reset
            ui.label(format!("Zoom: {}%", self.view_transform.zoom_percent()))
                .on_hover_text("Current zoom level. Scroll to zoom, drag to pan");
            if ui
                .button("Reset View")
                .on_hover_text("Reset zoom and pan to default")
                .clicked()
            {
                self.reset_view();
            }
        });

        // Check if we need to update texture
        let wl_key = (window_level.min, window_level.max);
        let needs_update = self.cached_slice_index != Some(self.current_slice)
            || self.cached_axis_mapping != Some(axis_mapping)
            || self.cached_colormap != Some(colormap)
            || self.cached_window_level != Some(wl_key)
            || self.texture_handle.is_none();

        if needs_update {
            self.update_texture(ui.ctx(), volume, colormap, window_level);
        }

        // Layout constants
        let colorbar_height = 30.0;
        let status_bar_height = 20.0;
        let available_size = ui.available_size();
        let image_area_height = available_size.y - colorbar_height - status_bar_height - 10.0;

        // Display image with pan/zoom
        if self.texture_handle.is_some() {
            let (img_width, img_height) = volume.display_dimensions();
            let img_aspect = img_width as f32 / img_height as f32;

            // Create a clip rect for the image area
            let (response, painter) = ui.allocate_painter(
                egui::vec2(available_size.x, image_area_height),
                egui::Sense::click_and_drag(),
            );

            let clip_rect = response.rect;

            // Calculate image rect based on aspect ratio setting and zoom
            let base_size = if keep_aspect_ratio {
                let panel_aspect = clip_rect.width() / clip_rect.height();
                if img_aspect > panel_aspect {
                    egui::vec2(clip_rect.width(), clip_rect.width() / img_aspect)
                } else {
                    egui::vec2(clip_rect.height() * img_aspect, clip_rect.height())
                }
            } else {
                egui::vec2(clip_rect.width(), clip_rect.height())
            };

            // Apply zoom
            let display_size = base_size * self.view_transform.zoom;

            // Calculate image position with pan offset
            let center = clip_rect.center() + self.view_transform.pan;
            let image_rect = egui::Rect::from_center_size(center, display_size);

            // Cache image rect and dimensions for cursor tracking
            self.cached_image_rect = Some(image_rect);
            self.cached_image_dims = Some((img_width, img_height));

            // Handle mouse interactions (pan/zoom and ROI)
            self.handle_mouse_input(
                &response,
                clip_rect.center(),
                image_rect,
                img_width,
                img_height,
            );

            // Track cursor position and compute image coordinates
            self.update_cursor_info(&response, image_rect, img_width, img_height, slice_data);

            // Draw image (clipped to available area)
            let clipped_painter = painter.with_clip_rect(clip_rect);
            if let Some(texture) = &self.texture_handle {
                clipped_painter.image(
                    texture.id(),
                    image_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            }

            // Draw ROI overlay (if any)
            self.draw_roi_overlay(&clipped_painter, image_rect, img_width, img_height);

            // Draw border around clip area
            painter.rect_stroke(clip_rect, 0.0, egui::Stroke::new(1.0, egui::Color32::GRAY));
        }

        // Cursor status bar
        self.draw_cursor_status_bar(ui, available_size.x, status_bar_height);

        // Horizontal colorbar below image
        self.draw_horizontal_colorbar(
            ui,
            available_size.x,
            colorbar_height,
            colormap,
            window_level,
        );

        slice_changed
    }

    /// Update cursor info based on hover position.
    fn update_cursor_info(
        &mut self,
        response: &egui::Response,
        image_rect: egui::Rect,
        img_width: usize,
        img_height: usize,
        slice_data: Option<&Array2<f32>>,
    ) {
        if let Some(hover_pos) = response.hover_pos() {
            // Check if cursor is within the image bounds
            if image_rect.contains(hover_pos) {
                // Convert screen position to normalized image coordinates [0, 1]
                let norm_x = (hover_pos.x - image_rect.left()) / image_rect.width();
                let norm_y = (hover_pos.y - image_rect.top()) / image_rect.height();

                // Convert to pixel coordinates
                let pixel_x = (norm_x * img_width as f32).floor() as i32;
                let pixel_y = (norm_y * img_height as f32).floor() as i32;

                // Bounds check
                if pixel_x >= 0
                    && pixel_x < img_width as i32
                    && pixel_y >= 0
                    && pixel_y < img_height as i32
                {
                    let x = pixel_x as usize;
                    let y = pixel_y as usize;

                    self.cursor_info.x = Some(x);
                    self.cursor_info.y = Some(y);

                    // Look up intensity if slice data is available
                    if let Some(data) = slice_data {
                        if y < data.nrows() && x < data.ncols() {
                            self.cursor_info.intensity = Some(data[[y, x]]);
                        } else {
                            self.cursor_info.intensity = None;
                        }
                    } else {
                        self.cursor_info.intensity = None;
                    }
                } else {
                    self.cursor_info.clear();
                }
            } else {
                self.cursor_info.clear();
            }
        } else {
            self.cursor_info.clear();
        }
    }

    /// Draw cursor status bar showing coordinates and intensity.
    fn draw_cursor_status_bar(&self, ui: &mut egui::Ui, width: f32, height: f32) {
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

                    if let Some(intensity) = self.cursor_info.intensity {
                        ui.label(
                            egui::RichText::new(format!("I: {:.4}", intensity))
                                .monospace()
                                .small(),
                        );
                    } else {
                        ui.label(egui::RichText::new("I: ---").monospace().small().weak());
                    }
                } else {
                    ui.label(
                        egui::RichText::new("(x: ---, y: ---)  I: ---")
                            .monospace()
                            .small()
                            .weak(),
                    );
                }
            });
        });
    }

    /// Draw horizontal colorbar with gradient and labels.
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

            for i in 0..num_steps {
                let [r, g, b] = lut[i];
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

            // Labels row
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(format!("{:.2}", window_level.min)).small());
                ui.add_space(width - 80.0);
                ui.label(egui::RichText::new(format!("{:.2}", window_level.max)).small());
            });
        });
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

    fn handle_mouse_input(
        &mut self,
        response: &egui::Response,
        center: egui::Pos2,
        image_rect: egui::Rect,
        img_width: usize,
        img_height: usize,
    ) {
        // Check if Shift is held for ROI drawing mode
        let shift_held = response.ctx.input(|i| i.modifiers.shift);

        // Determine if we should draw ROI: either roi_mode is ON or Shift is held
        let should_draw_roi = self.roi_mode || shift_held;

        // Handle scroll wheel for zoom (always active)
        if response.hovered() {
            let scroll_delta = response.ctx.input(|i| i.raw_scroll_delta.y);
            if scroll_delta != 0.0 {
                let zoom_factor = 1.0 + scroll_delta * 0.001;
                let new_zoom = (self.view_transform.zoom * zoom_factor).clamp(0.1, 10.0);

                // Zoom centered on cursor
                if let Some(pointer_pos) = response.hover_pos() {
                    let cursor_offset = pointer_pos - center - self.view_transform.pan;
                    let zoom_ratio = new_zoom / self.view_transform.zoom;
                    self.view_transform.pan -= cursor_offset * (zoom_ratio - 1.0);
                }

                self.view_transform.zoom = new_zoom;
            }

            // Update cursor_in_roi for visual feedback
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
                        } else {
                            // Normal pan mode (no ROI mode, outside ROI)
                            if let Some(last_pos) = self.last_drag_pos {
                                let delta = pointer_pos - last_pos;
                                self.view_transform.pan += delta;
                            }
                            self.last_drag_pos = Some(pointer_pos);
                            self.is_dragging = true;
                        }
                    }
                } else {
                    // Outside image - allow panning
                    if !self.roi_state.drawing && !self.roi_state.moving {
                        if let Some(last_pos) = self.last_drag_pos {
                            let delta = pointer_pos - last_pos;
                            self.view_transform.pan += delta;
                        }
                        self.last_drag_pos = Some(pointer_pos);
                        self.is_dragging = true;
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

    /// Draw ROI overlay on the image.
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

    fn update_texture(
        &mut self,
        ctx: &egui::Context,
        volume: &Volume3D,
        colormap: Colormap,
        window_level: &WindowLevel,
    ) {
        if let Some(slice) = volume.get_slice(self.current_slice) {
            let (height, width) = (slice.nrows(), slice.ncols());

            // Generate LUT for colormap
            let lut = colormap.generate_lut();

            // Convert to RGBA with window/level and colormap
            let rgba_pixels: Vec<u8> = slice
                .iter()
                .flat_map(|&val| {
                    let normalized = window_level.normalize(val);
                    let lut_index = (normalized * 255.0) as usize;
                    let [r, g, b] = lut[lut_index.min(255)];
                    [r, g, b, 255]
                })
                .collect();

            let color_image =
                egui::ColorImage::from_rgba_unmultiplied([width, height], &rgba_pixels);

            let texture_options = egui::TextureOptions {
                magnification: egui::TextureFilter::Nearest,
                minification: egui::TextureFilter::Linear,
                ..Default::default()
            };

            self.texture_handle =
                Some(ctx.load_texture("slice_texture", color_image, texture_options));

            self.cached_slice_index = Some(self.current_slice);
            self.cached_axis_mapping = Some(volume.axis_mapping());
            self.cached_colormap = Some(colormap);
            self.cached_window_level = Some((window_level.min, window_level.max));
        }
    }
}
