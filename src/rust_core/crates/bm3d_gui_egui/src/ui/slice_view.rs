use crate::data::{AxisMapping, Volume3D};
use crate::ui::colormap::Colormap;
use crate::ui::window_level::WindowLevel;
use eframe::egui;

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
            if ui.button("◄").clicked() && self.current_slice > 0 {
                self.current_slice -= 1;
                slice_changed = true;
            }

            // Slider
            let slider_response = ui.add(
                egui::Slider::new(&mut self.current_slice, 0..=num_slices.saturating_sub(1))
                    .text("")
                    .show_value(false),
            );
            if slider_response.changed() {
                slice_changed = true;
            }

            // Step forward button
            if ui.button("►").clicked() && self.current_slice < num_slices - 1 {
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
            ui.label(format!("Zoom: {}%", self.view_transform.zoom_percent()));
            if ui.button("Reset View").clicked() {
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
        let available_size = ui.available_size();
        let image_area_height = available_size.y - colorbar_height - 5.0;

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

            // Handle mouse interactions
            self.handle_mouse_input(&response, clip_rect.center());

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

        // Horizontal colorbar below image
        self.draw_horizontal_colorbar(ui, available_size.x, colorbar_height, colormap, window_level);

        slice_changed
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
