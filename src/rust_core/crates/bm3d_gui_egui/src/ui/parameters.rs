use bm3d_core::{Bm3dConfig, RingRemovalMode};
use eframe::egui;

/// BM3D processing parameters with two-tier UI.
pub struct Bm3dParameters {
    // Tier 1 - Simple (always visible)
    pub mode: RingRemovalMode,
    /// Random noise standard deviation (maps to Rust core's sigma_random).
    pub sigma_random: f32,

    // Tier 2 - Advanced (behind toggle)
    pub patch_size: usize,
    pub search_window: usize,
    pub max_matches: usize,
    /// Which axis to iterate over for sinogram processing (0, 1, or 2).
    /// Default is 1 (Y axis) for standard tomography data [angles, Y, X].
    pub processing_axis: usize,

    // UI state
    show_advanced: bool,
}

impl Default for Bm3dParameters {
    fn default() -> Self {
        // GUI defaults optimized for neutron sinogram data:
        // - sigma_random: 0.005 (vs Rust default 0.1) - typical noise level for neutron imaging
        // - max_matches: 32 (vs Rust default 16) - better quality for interactive use
        Self {
            mode: RingRemovalMode::Streak,
            sigma_random: 0.005,
            patch_size: 8,
            search_window: 24,
            max_matches: 32,
            processing_axis: 1, // Default: Y axis (middle dimension)
            show_advanced: false,
        }
    }
}

impl Bm3dParameters {
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert to bm3d_core config.
    pub fn to_config(&self) -> Bm3dConfig<f32> {
        let mut config = Bm3dConfig::default();
        config.sigma_random = self.sigma_random;
        config.patch_size = self.patch_size;
        config.step_size = self.patch_size / 2; // Standard: half patch size
        config.search_window = self.search_window;
        config.max_matches = self.max_matches;
        config
    }

    /// Show parameter controls. Returns true if any parameter changed.
    pub fn show(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        ui.heading("BM3D Parameters");

        // Tier 1 - Simple parameters (always visible)
        ui.horizontal(|ui| {
            ui.label("Mode:")
                .on_hover_text("Generic: Standard denoising for white noise\nStreak: Optimized for ring artifact removal in sinograms");

            egui::ComboBox::from_id_salt("bm3d_mode")
                .selected_text(match self.mode {
                    RingRemovalMode::Generic => "Generic",
                    RingRemovalMode::Streak => "Streak",
                })
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_value(&mut self.mode, RingRemovalMode::Generic, "Generic")
                        .changed()
                    {
                        changed = true;
                    }
                    if ui
                        .selectable_value(&mut self.mode, RingRemovalMode::Streak, "Streak")
                        .changed()
                    {
                        changed = true;
                    }
                });
        });

        ui.horizontal(|ui| {
            ui.label("Sigma:")
                .on_hover_text("Noise level estimate (sigma_random). Higher values = stronger denoising.\nTypical range: 0.001 - 0.5\nSupports scientific notation (e.g., 5e-3)");

            // Use DragValue which accepts scientific notation input (e.g., 5e-3, 1.5E-4)
            let sigma_response = ui.add(
                egui::DragValue::new(&mut self.sigma_random)
                    .speed(0.0001)
                    .range(0.0001..=0.5)
                    .max_decimals(4),
            );
            if sigma_response.changed() {
                changed = true;
            }
        });

        ui.add_space(5.0);

        // Tier 2 - Advanced parameters (behind toggle)
        ui.horizontal(|ui| {
            if ui
                .selectable_label(self.show_advanced, "⚙ Advanced")
                .clicked()
            {
                self.show_advanced = !self.show_advanced;
            }
        });

        if self.show_advanced {
            ui.indent("advanced_params", |ui| {
                // Patch size
                ui.horizontal(|ui| {
                    ui.label("Patch Size:")
                        .on_hover_text("Size of patches for block matching.\nLarger = smoother but slower.\nSmaller = preserves fine details.");

                    egui::ComboBox::from_id_salt("patch_size")
                        .selected_text(format!("{}", self.patch_size))
                        .show_ui(ui, |ui| {
                            for size in [4, 8, 16] {
                                if ui
                                    .selectable_value(&mut self.patch_size, size, format!("{}", size))
                                    .changed()
                                {
                                    changed = true;
                                }
                            }
                        });
                });

                // Search window
                ui.horizontal(|ui| {
                    ui.label("Search Window:")
                        .on_hover_text("Size of area to search for similar patches.\nLarger = better matches but slower.\nRange: 16-64");

                    let sw_response = ui.add(
                        egui::Slider::new(&mut self.search_window, 16..=64).step_by(4.0),
                    );
                    if sw_response.changed() {
                        changed = true;
                    }
                });

                // Max matches
                ui.horizontal(|ui| {
                    ui.label("Max Matches:")
                        .on_hover_text("Maximum similar patches per group.\nMore = better denoising but slower.\nRange: 8-64");

                    let mm_response = ui.add(
                        egui::Slider::new(&mut self.max_matches, 8..=64).step_by(4.0),
                    );
                    if mm_response.changed() {
                        changed = true;
                    }
                });

                // Processing axis
                ui.horizontal(|ui| {
                    ui.label("Process Axis:")
                        .on_hover_text("Which dimension to iterate over for processing.\nFor [angles, Y, X] data:\n  • Axis 0: Process [Y, X] slices (unusual)\n  • Axis 1: Process [angles, X] sinograms (default)\n  • Axis 2: Process [angles, Y] slices (unusual)");

                    egui::ComboBox::from_id_salt("processing_axis")
                        .selected_text(format!("Axis {} (D{})", self.processing_axis, self.processing_axis))
                        .show_ui(ui, |ui| {
                            for axis in 0..3 {
                                let label = match axis {
                                    0 => "Axis 0 (D0)",
                                    1 => "Axis 1 (D1) - Default",
                                    2 => "Axis 2 (D2)",
                                    _ => unreachable!(),
                                };
                                if ui
                                    .selectable_value(&mut self.processing_axis, axis, label)
                                    .changed()
                                {
                                    changed = true;
                                }
                            }
                        });
                });

                // Reset to defaults button
                if ui.button("Reset to Defaults").clicked() {
                    self.patch_size = 8;
                    self.search_window = 24;
                    self.max_matches = 32;
                    self.processing_axis = 1;
                    changed = true;
                }
            });
        }

        changed
    }
}
