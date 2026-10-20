use crate::data::Volume3D;
use crate::ui::colormap::{Colormap, DivergingColormap};
use crate::ui::histogram::colors;
use crate::ui::slice_view::{ImageRoi, RoiState};
use crate::ui::window_level::WindowLevel;
use eframe::egui;
use ndarray::Array2;

#[derive(Clone, Default)]
pub struct CompareCursorInfo {
    /// Image X coordinate (column)
    pub x: Option<usize>,
    /// Image Y coordinate (row)
    pub y: Option<usize>,
    pub orig_intensity: Option<f32>,
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
    current_slice: usize,

    original_texture: Option<egui::TextureHandle>,
    processed_texture: Option<egui::TextureHandle>,
    difference_texture: Option<egui::TextureHandle>,

    cached_slice_index: Option<usize>,
    cached_colormap: Option<Colormap>,
    cached_window_level: Option<(f32, f32)>,

    // Difference range (symmetric around zero)
    diff_max: f32,

    cursor_info: CompareCursorInfo,

    // ROI selection (shared across all three panels)
    roi_state: RoiState,
    /// ROI drawing mode active (drag draws ROI without Shift).
    pub roi_mode: bool,
    cursor_in_roi: bool,

    is_dragging: bool,
    last_drag_pos: Option<egui::Pos2>,

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

    pub fn is_cursor_in_roi(&self) -> bool {
        self.cursor_in_roi
    }

    pub fn roi(&self) -> Option<&ImageRoi> {
        self.roi_state.roi.as_ref()
    }

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

        if self.current_slice >= num_slices {
            self.current_slice = num_slices.saturating_sub(1);
        }

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

        let wl_key = (window_level.min, window_level.max);
        let needs_update = self.cached_slice_index != Some(self.current_slice)
            || self.cached_colormap != Some(colormap)
            || self.cached_window_level != Some(wl_key)
            || self.original_texture.is_none();

        if needs_update {
            self.update_textures(ui.ctx(), original, processed, colormap, window_level);
        }

        let (img_width, img_height) = original.display_dimensions();
        let img_aspect = img_width as f32 / img_height as f32;

        let available_size = ui.available_size();
        let colorbar_height = 30.0;
        let label_height = 20.0;
        let status_bar_height = 20.0;
        let spacing = 5.0;

        let panel_width = (available_size.x - 2.0 * spacing) / 3.0;
        let image_height =
            available_size.y - colorbar_height - label_height - status_bar_height - 3.0 * spacing;

        ui.horizontal(|ui| {
            ui.add_sized(
                egui::vec2(panel_width, label_height),
                egui::Label::new(egui::RichText::new("Original").strong()),
            );
            ui.add_space(spacing);
            ui.add_sized(
                egui::vec2(panel_width, label_height),
                egui::Label::new(egui::RichText::new("Processed").strong()),
            );
            ui.add_space(spacing);
            ui.add_sized(
                egui::vec2(panel_width, label_height),
                egui::Label::new(egui::RichText::new("Difference").strong()),
            );
        });

        self.cursor_info.clear();

        let mut panel_data: Vec<(egui::Response, Option<egui::Rect>, egui::Painter)> = Vec::new();

        ui.horizontal(|ui| {
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

        self.cached_image_dims = Some((img_width, img_height));

        let mut active_response: Option<&egui::Response> = None;
        let mut active_image_rect: Option<egui::Rect> = None;

        for (response, image_rect, _) in &panel_data {
            if let Some(rect) = image_rect
                && (response.hovered() || response.dragged())
            {
                active_response = Some(response);
                active_image_rect = Some(*rect);
                self.cached_image_rect = Some(*rect);
                break;
            }
        }

        if let (Some(response), Some(image_rect)) = (active_response, active_image_rect) {
            let response_clone = response.clone();
            self.handle_mouse_input(&response_clone, image_rect, img_width, img_height);
        } else {
            self.cursor_in_roi = false;
            if self.roi_state.drawing {
                self.roi_state.finish_draw();
            }
            if self.roi_state.moving {
                self.roi_state.finish_move();
            }
        }

        for (response, image_rect, _) in &panel_data {
            if let Some((x, y)) = self.get_image_coords(response, image_rect, img_width, img_height)
            {
                self.cursor_info.x = Some(x);
                self.cursor_info.y = Some(y);

                if let Some(orig_data) = orig_slice
                    && y < orig_data.nrows()
                    && x < orig_data.ncols()
                {
                    self.cursor_info.orig_intensity = Some(orig_data[[y, x]]);
                }
                if let Some(proc_data) = proc_slice
                    && y < proc_data.nrows()
                    && x < proc_data.ncols()
                {
                    self.cursor_info.proc_intensity = Some(proc_data[[y, x]]);
                }
                if let (Some(orig_val), Some(proc_val)) = (
                    self.cursor_info.orig_intensity,
                    self.cursor_info.proc_intensity,
                ) {
                    self.cursor_info.diff_intensity = Some(orig_val - proc_val);
                }
                break;
            }
        }

        for (_, image_rect, painter) in &panel_data {
            if let Some(rect) = image_rect {
                self.draw_roi_overlay(painter, *rect, img_width, img_height);
            }
        }

        self.draw_cursor_status_bar(ui, available_size.x, status_bar_height);

        ui.horizontal(|ui| {
            let shared_colorbar_width = panel_width * 2.0 + spacing;
            self.draw_horizontal_colorbar(
                ui,
                shared_colorbar_width,
                colorbar_height,
                colormap,
                window_level,
            );
            ui.add_space(spacing);

            self.draw_horizontal_diff_colorbar(ui, panel_width, colorbar_height);
        });

        slice_changed
    }

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

        let norm_x = (hover_pos.x - rect.left()) / rect.width();
        let norm_y = (hover_pos.y - rect.top()) / rect.height();

        let pixel_x = (norm_x * img_width as f32).floor() as i32;
        let pixel_y = (norm_y * img_height as f32).floor() as i32;

        if pixel_x >= 0 && pixel_x < img_width as i32 && pixel_y >= 0 && pixel_y < img_height as i32
        {
            Some((pixel_x as usize, pixel_y as usize))
        } else {
            None
        }
    }

    fn draw_cursor_status_bar(&self, ui: &mut egui::Ui, width: f32, height: f32) {
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

        painter.rect_stroke(
            response.rect,
            0.0,
            egui::Stroke::new(1.0_f32, egui::Color32::GRAY),
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

            painter.rect_stroke(rect, 0.0, egui::Stroke::new(1.0_f32, egui::Color32::GRAY));

            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(format!("{:.2}", window_level.min)).small());
                ui.add_space(width - 80.0);
                ui.label(egui::RichText::new(format!("{:.2}", window_level.max)).small());
            });
        });
    }

    fn draw_horizontal_diff_colorbar(&self, ui: &mut egui::Ui, width: f32, height: f32) {
        ui.vertical(|ui| {
            let (response, painter) =
                ui.allocate_painter(egui::vec2(width, height - 15.0), egui::Sense::hover());

            let rect = response.rect;
            let num_steps = 256;
            let step_width = rect.width() / num_steps as f32;

            for i in 0..num_steps {
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

            painter.rect_stroke(rect, 0.0, egui::Stroke::new(1.0_f32, egui::Color32::GRAY));

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

            let diff: Array2<f32> = &orig - &proc;

            let mut max_abs = 0.0f32;
            for &val in diff.iter() {
                if val.is_finite() {
                    max_abs = max_abs.max(val.abs());
                }
            }
            self.diff_max = if max_abs > 0.0 { max_abs } else { 1.0 };

            let rdbu_lut = DivergingColormap::generate_lut();
            let diff_rgba: Vec<u8> = diff
                .iter()
                .flat_map(|&val| {
                    let normalized = if self.diff_max > 0.0 {
                        val / self.diff_max
                    } else {
                        0.0
                    };
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

            self.cached_slice_index = Some(self.current_slice);
            self.cached_colormap = Some(colormap);
            self.cached_window_level = Some((window_level.min, window_level.max));
        }
    }

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

        let norm_x = (screen_pos.x - image_rect.left()) / image_rect.width();
        let norm_y = (screen_pos.y - image_rect.top()) / image_rect.height();

        let pixel_x = (norm_x * img_width as f32).floor() as i32;
        let pixel_y = (norm_y * img_height as f32).floor() as i32;

        if pixel_x >= 0 && pixel_x < img_width as i32 && pixel_y >= 0 && pixel_y < img_height as i32
        {
            Some((pixel_x as usize, pixel_y as usize))
        } else {
            None
        }
    }

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
        image_rect: egui::Rect,
        img_width: usize,
        img_height: usize,
    ) {
        let shift_held = response.ctx.input(|i| i.modifiers.shift);

        let should_draw_roi = self.roi_mode || shift_held;

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

        if response.dragged() {
            if let Some(pointer_pos) = response.interact_pointer_pos()
                && let Some((img_x, img_y)) =
                    self.screen_to_image(pointer_pos, image_rect, img_width, img_height)
            {
                if self.roi_state.drawing {
                    self.roi_state.update_draw(img_x, img_y);
                    self.is_dragging = false;
                    self.last_drag_pos = None;
                } else if self.roi_state.moving {
                    self.roi_state
                        .update_move(img_x, img_y, img_width, img_height);
                    self.is_dragging = false;
                    self.last_drag_pos = None;
                } else {
                    let inside_roi = self.roi_state.cursor_inside_roi(img_x, img_y);

                    if inside_roi {
                        self.roi_state.start_move(img_x, img_y);
                        self.is_dragging = false;
                        self.last_drag_pos = None;
                    } else if should_draw_roi {
                        self.roi_state.start_draw(img_x, img_y);
                        self.is_dragging = false;
                        self.last_drag_pos = None;
                    }
                }
            }
        } else {
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

    fn draw_roi_overlay(
        &self,
        painter: &egui::Painter,
        image_rect: egui::Rect,
        img_width: usize,
        img_height: usize,
    ) {
        let roi_to_draw = self.roi_state.drawing_rect().or(self.roi_state.roi);

        if let Some(roi) = roi_to_draw {
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

            let fill_alpha = if self.cursor_in_roi && !self.roi_state.drawing {
                60
            } else {
                30
            };
            let fill_color = egui::Color32::from_rgba_unmultiplied(255, 255, 0, fill_alpha);
            painter.rect_filled(screen_rect, 0.0, fill_color);

            let stroke_color = if self.roi_state.drawing {
                egui::Color32::from_rgb(0, 255, 255)
            } else if self.roi_state.moving {
                egui::Color32::from_rgb(0, 255, 0)
            } else if self.cursor_in_roi {
                egui::Color32::from_rgb(255, 165, 0)
            } else {
                egui::Color32::from_rgb(255, 255, 0)
            };
            let stroke_width = if self.cursor_in_roi || self.roi_state.moving {
                3.0_f32
            } else {
                2.0_f32
            };
            painter.rect_stroke(
                screen_rect,
                0.0,
                egui::Stroke::new(stroke_width, stroke_color),
            );
        }
    }
}
