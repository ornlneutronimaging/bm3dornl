use crate::data::{AxisMapping, Volume3D};
use crate::ui::colormap::Colormap;
use crate::ui::window_level::WindowLevel;
use eframe::egui;
use ndarray::Array2;

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
            if ui.button("◄")
                .on_hover_text("Previous slice")
                .clicked() && self.current_slice > 0 {
                self.current_slice -= 1;
                slice_changed = true;
            }

            // Slider
            let slider_response = ui.add(
                egui::Slider::new(&mut self.current_slice, 0..=num_slices.saturating_sub(1))
                    .text("")
                    .show_value(false),
            ).on_hover_text("Navigate through slices along the selected axis");
            if slider_response.changed() {
                slice_changed = true;
            }

            // Step forward button
            if ui.button("►")
                .on_hover_text("Next slice")
                .clicked() && self.current_slice < num_slices - 1 {
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
            if ui.button("Reset View")
                .on_hover_text("Reset zoom and pan to default")
                .clicked() {
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
            let (response, painter) =
                ui.allocate_painter(egui::vec2(available_size.x, image_area_height), egui::Sense::click_and_drag());

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

            // Handle mouse interactions
            self.handle_mouse_input(&response, clip_rect.center());

            // Track cursor position and compute image coordinates
            self.update_cursor_info(&response, image_rect, img_width, img_height, slice_data);

            // Draw image (clipped to available area)
            if let Some(texture) = &self.texture_handle {
                painter.with_clip_rect(clip_rect).image(
                    texture.id(),
                    image_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            }

            // Draw border around clip area
            painter.rect_stroke(
                clip_rect,
                0.0,
                egui::Stroke::new(1.0, egui::Color32::GRAY),
            );
        }

        // Cursor status bar
        self.draw_cursor_status_bar(ui, available_size.x, status_bar_height);

        // Horizontal colorbar below image
        self.draw_horizontal_colorbar(ui, available_size.x, colorbar_height, colormap, window_level);

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
                if pixel_x >= 0 && pixel_x < img_width as i32 && pixel_y >= 0 && pixel_y < img_height as i32 {
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
                        ui.label(
                            egui::RichText::new("I: ---")
                                .monospace()
                                .small()
                                .weak(),
                        );
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
            let (response, painter) = ui.allocate_painter(
                egui::vec2(width, height - 15.0),
                egui::Sense::hover(),
            );

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

    fn handle_mouse_input(&mut self, response: &egui::Response, center: egui::Pos2) {
        // Handle scroll wheel for zoom
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
        }

        // Handle drag for pan
        if response.dragged() {
            if let Some(pointer_pos) = response.interact_pointer_pos() {
                if let Some(last_pos) = self.last_drag_pos {
                    let delta = pointer_pos - last_pos;
                    self.view_transform.pan += delta;
                }
                self.last_drag_pos = Some(pointer_pos);
                self.is_dragging = true;
            }
        } else {
            self.is_dragging = false;
            self.last_drag_pos = None;
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

            self.texture_handle = Some(ctx.load_texture("slice_texture", color_image, texture_options));

            self.cached_slice_index = Some(self.current_slice);
            self.cached_axis_mapping = Some(volume.axis_mapping());
            self.cached_colormap = Some(colormap);
            self.cached_window_level = Some((window_level.min, window_level.max));
        }
    }
}
