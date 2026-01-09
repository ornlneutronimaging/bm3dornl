use crate::data::Volume3D;
use crate::ui::colormap::{Colormap, DivergingColormap};
use crate::ui::window_level::WindowLevel;
use eframe::egui;
use ndarray::Array2;

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
        let num_slices = original.num_slices();
        let mut slice_changed = false;

        // Clamp current slice
        if self.current_slice >= num_slices {
            self.current_slice = num_slices.saturating_sub(1);
        }

        // Navigation controls (shared for all panels)
        ui.horizontal(|ui| {
            if ui.button("◄")
                .on_hover_text("Previous slice")
                .clicked() && self.current_slice > 0 {
                self.current_slice -= 1;
                slice_changed = true;
            }

            let slider_response = ui.add(
                egui::Slider::new(&mut self.current_slice, 0..=num_slices.saturating_sub(1))
                    .text("")
                    .show_value(false),
            ).on_hover_text("Navigate through slices along the selected axis");
            if slider_response.changed() {
                slice_changed = true;
            }

            if ui.button("►")
                .on_hover_text("Next slice")
                .clicked() && self.current_slice < num_slices - 1 {
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
        let spacing = 5.0;

        // Calculate panel dimensions
        let panel_width = (available_size.x - 2.0 * spacing) / 3.0;
        let image_height = available_size.y - colorbar_height - label_height - 2.0 * spacing;

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

        // Row 2: Images
        ui.horizontal(|ui| {
            // Original image
            self.draw_image_panel(ui, &self.original_texture, panel_width, image_height, img_aspect, keep_aspect_ratio);
            ui.add_space(spacing);

            // Processed image
            self.draw_image_panel(ui, &self.processed_texture, panel_width, image_height, img_aspect, keep_aspect_ratio);
            ui.add_space(spacing);

            // Difference image
            self.draw_image_panel(ui, &self.difference_texture, panel_width, image_height, img_aspect, keep_aspect_ratio);
        });

        // Row 3: Colorbars
        ui.horizontal(|ui| {
            // Shared colorbar for Original and Processed (spans 2 panels + spacing)
            let shared_colorbar_width = panel_width * 2.0 + spacing;
            self.draw_horizontal_colorbar(ui, shared_colorbar_width, colorbar_height, colormap, window_level);
            ui.add_space(spacing);

            // Difference colorbar (RdBu)
            self.draw_horizontal_diff_colorbar(ui, panel_width, colorbar_height);
        });

        slice_changed
    }

    fn draw_image_panel(
        &self,
        ui: &mut egui::Ui,
        texture: &Option<egui::TextureHandle>,
        width: f32,
        height: f32,
        img_aspect: f32,
        keep_aspect_ratio: bool,
    ) {
        let (response, painter) = ui.allocate_painter(
            egui::vec2(width, height),
            egui::Sense::hover(),
        );

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

    fn draw_horizontal_diff_colorbar(&self, ui: &mut egui::Ui, width: f32, height: f32) {
        ui.vertical(|ui| {
            // Colorbar gradient (RdBu: blue -> white -> red)
            let (response, painter) = ui.allocate_painter(
                egui::vec2(width, height - 15.0),
                egui::Sense::hover(),
            );

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
}
