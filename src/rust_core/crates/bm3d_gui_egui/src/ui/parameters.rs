use bm3d_core::{Bm3dConfig, MultiscaleConfig, RingRemovalMode};
use eframe::egui;

/// BM3D processing parameters with two-tier UI.
pub struct Bm3dParameters {
    // Tier 1 - Simple (always visible)
    pub mode: RingRemovalMode,
    /// Random noise standard deviation (maps to Rust core's sigma_random).
    pub sigma_random: f32,
    /// Automatically estimate sigma from image data
    pub auto_sigma: bool,

    // Tier 2 - Advanced (behind toggle)
    pub patch_size: usize,
    pub search_window: usize,
    pub max_matches: usize,
    /// Which axis to iterate over for sinogram processing (0, 1, or 2).
    /// Default is 1 (Y axis) for standard tomography data [angles, Y, X].
    pub processing_axis: usize,

    // SVD-MG parameters
    pub fft_alpha: f32,
    pub notch_width: f32,

    // Multi-scale parameters
    pub multiscale: bool,
    pub num_scales: usize, // 0 = auto

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
            auto_sigma: true, // Default to auto
            patch_size: 8,
            search_window: 24,
            max_matches: 32,
            processing_axis: 1, // Default: Y axis (middle dimension)
            fft_alpha: 1.0,
            notch_width: 2.0,
            multiscale: false,
            num_scales: 0,
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
        let default = Bm3dConfig::<f32>::default();
        // If auto_sigma is enabled, we pass 0.0 to Rust core to trigger auto-estimation.
        // Otherwise we pass the manual value.
        let sigma = if self.auto_sigma {
            0.0
        } else {
            self.sigma_random
        };

        Bm3dConfig {
            sigma_random: sigma,
            patch_size: self.patch_size,
            step_size: self.patch_size / 2, // Standard: half patch size
            search_window: self.search_window,
            max_matches: self.max_matches,
            fft_alpha: self.fft_alpha,
            notch_width: self.notch_width,
            ..default
        }
    }

    /// Convert to multiscale config
    pub fn to_multiscale_config(&self) -> MultiscaleConfig<f32> {
        let bm3d_config = self.to_config();
        let default = MultiscaleConfig::<f32>::default();

        MultiscaleConfig {
            bm3d_config,
            num_scales: if self.num_scales > 0 {
                Some(self.num_scales)
            } else {
                None
            },
            ..default
        }
    }

    /// Show parameter controls. Returns true if any parameter changed.
    pub fn show(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        ui.heading("BM3D Parameters");

        // Tier 1 - Simple parameters (always visible)
        ui.horizontal(|ui| {
            ui.label("Mode:")
                .on_hover_text("Generic: Standard denoising for white noise\nStreak: Optimized for ring artifact removal\nFourier-SVD: Fast FFT-guided SVD destriping for subtle artifacts");

            egui::ComboBox::from_id_salt("bm3d_mode")
                .selected_text(match self.mode {
                    RingRemovalMode::Generic => "Generic",
                    RingRemovalMode::Streak => "Streak",
                    RingRemovalMode::FourierSvd => "Fourier-SVD",
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
                    if ui
                        .selectable_value(&mut self.mode, RingRemovalMode::FourierSvd, "Fourier-SVD")
                        .changed()
                    {
                        changed = true;
                    }
                });
        });

        // Show Fourier-SVD specific parameters
        if self.mode == RingRemovalMode::FourierSvd {
            ui.horizontal(|ui| {
                ui.label("FFT Alpha:")
                    .on_hover_text("FFT Trust Factor (0.0 - 5.0). 1.0 = standard.");
                if ui
                    .add(egui::Slider::new(&mut self.fft_alpha, 0.0..=5.0).step_by(0.1))
                    .changed()
                {
                    changed = true;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Notch Width:")
                    .on_hover_text("Selectivity of vertical frequency notch filter.");
                if ui
                    .add(egui::Slider::new(&mut self.notch_width, 0.5..=5.0).step_by(0.1))
                    .changed()
                {
                    changed = true;
                }
            });
        }

        ui.horizontal(|ui| {
            ui.label("Sigma:")
                .on_hover_text("Noise level estimate (sigma_random). Higher values = stronger denoising.\nTypical range: 0.001 - 0.5\nSupports scientific notation (e.g., 5e-3)");

            // Auto sigma checkbox
            if ui.checkbox(&mut self.auto_sigma, "Auto")
                .on_hover_text("Automatically estimate noise level from image data")
                .changed() {
                changed = true;
            }

            // Disable drag value if auto is selected
            ui.add_enabled_ui(!self.auto_sigma, |ui| {
                // Use DragValue which accepts scientific notation input (e.g., 5e-3, 1.5E-4)
                let sigma_response = ui.add(
                    egui::DragValue::new(&mut self.sigma_random)
                        .speed(0.0001)
                        .range(0.0..=0.5) // Allow 0.0 manual input (though auto overrides if checked)
                        .max_decimals(4),
                );
                if sigma_response.changed() {
                    changed = true;
                }
            });
        });

        ui.add_space(5.0);

        // Tier 2 - Advanced parameters (behind toggle)
        ui.horizontal(|ui| {
            if ui
                .selectable_label(self.show_advanced, "⚙ Advanced")
                .on_hover_text("Show advanced parameters for fine-tuning BM3D algorithm")
                .clicked()
            {
                self.show_advanced = !self.show_advanced;
            }
        });

        if self.show_advanced {
            ui.indent("advanced_params", |ui| {
                // Multi-scale toggle
                ui.horizontal(|ui| {
                        ui.label("Multi-Scale:")
                        .on_hover_text("Use multi-scale algorithm to remove wide streaks.");
                    if ui.checkbox(&mut self.multiscale, "Enable").changed() {
                        changed = true;
                    }
                });

                if self.multiscale {
                    ui.indent("multiscale_params", |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Scales:")
                                .on_hover_text("Number of scales (0 = Auto).");
                            if ui.add(egui::DragValue::new(&mut self.num_scales).range(0..=6)).changed() {
                                changed = true;
                            }
                        });
                    });
                }

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
                if ui.button("Reset to Defaults")
                    .on_hover_text("Reset all advanced parameters to their default values")
                    .clicked() {
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
