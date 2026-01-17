use bm3d_core::{Bm3dConfig, MultiscaleConfig, RingRemovalMode};
use eframe::egui;

/// BM3D processing parameters with method-specific UI panels.
pub struct Bm3dParameters {
    // Mode selection
    pub mode: RingRemovalMode,

    // Common parameters
    /// Random noise standard deviation (maps to Rust core's sigma_random).
    pub sigma_random: f32,
    /// Automatically estimate sigma from image data
    pub auto_sigma: bool,

    // BM3D block-matching parameters (Generic, Streak, MultiscaleStreak)
    pub patch_size: usize,
    pub search_window: usize,
    pub max_matches: usize,
    /// Which axis to iterate over for sinogram processing (0, 1, or 2).
    /// Default is 1 (Y axis) for standard tomography data [angles, Y, X].
    pub processing_axis: usize,

    // Fourier-SVD parameters
    pub fft_alpha: f32,
    pub notch_width: f32,

    // Multi-scale parameters (MultiscaleStreak mode)
    pub num_scales: usize, // 0 = auto

    // UI state
    show_advanced: bool,
}

impl Default for Bm3dParameters {
    fn default() -> Self {
        // GUI defaults optimized for neutron sinogram data:
        // - MultiscaleStreak as default mode for best quality
        // - sigma_random: 0.005 (vs Rust default 0.1) - typical noise level for neutron imaging
        // - max_matches: 32 (vs Rust default 16) - better quality for interactive use
        Self {
            mode: RingRemovalMode::MultiscaleStreak,
            sigma_random: 0.005,
            auto_sigma: true, // Default to auto
            patch_size: 8,
            search_window: 24,
            max_matches: 32,
            processing_axis: 1, // Default: Y axis (middle dimension)
            fft_alpha: 1.0,
            notch_width: 2.0,
            num_scales: 0, // 0 = auto
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

    /// Returns true if the current mode uses multiscale processing
    pub fn uses_multiscale(&self) -> bool {
        self.mode == RingRemovalMode::MultiscaleStreak
    }

    /// Show parameter controls. Returns true if any parameter changed.
    pub fn show(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        ui.heading("Processing Parameters");

        // Mode selection (always visible)
        changed |= self.show_mode_selection(ui);

        ui.add_space(4.0);
        ui.separator();
        ui.add_space(4.0);

        // Method-specific parameter panels
        match self.mode {
            RingRemovalMode::FourierSvd => {
                changed |= self.show_fourier_svd_params(ui);
            }
            RingRemovalMode::MultiscaleStreak => {
                changed |= self.show_multiscale_streak_params(ui);
            }
            RingRemovalMode::Streak | RingRemovalMode::Generic => {
                changed |= self.show_bm3d_params(ui);
            }
        }

        changed
    }

    /// Show mode selection dropdown
    fn show_mode_selection(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        ui.horizontal(|ui| {
            ui.label("Method:").on_hover_text(
                "Multiscale Streak: Best quality for wide ring artifacts (default)\n\
                 Streak: Single-scale BM3D for narrow streaks\n\
                 Generic: Standard denoising for white noise\n\
                 Fourier-SVD: Fast FFT-guided SVD for subtle artifacts",
            );

            egui::ComboBox::from_id_salt("bm3d_mode")
                .selected_text(match self.mode {
                    RingRemovalMode::MultiscaleStreak => "Multiscale Streak",
                    RingRemovalMode::Streak => "Streak",
                    RingRemovalMode::Generic => "Generic",
                    RingRemovalMode::FourierSvd => "Fourier-SVD",
                })
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_value(
                            &mut self.mode,
                            RingRemovalMode::MultiscaleStreak,
                            "Multiscale Streak",
                        )
                        .on_hover_text("Best quality for wide ring artifacts")
                        .changed()
                    {
                        changed = true;
                    }
                    if ui
                        .selectable_value(&mut self.mode, RingRemovalMode::Streak, "Streak")
                        .on_hover_text("Single-scale BM3D for narrow streaks")
                        .changed()
                    {
                        changed = true;
                    }
                    if ui
                        .selectable_value(&mut self.mode, RingRemovalMode::Generic, "Generic")
                        .on_hover_text("Standard BM3D denoising")
                        .changed()
                    {
                        changed = true;
                    }
                    if ui
                        .selectable_value(
                            &mut self.mode,
                            RingRemovalMode::FourierSvd,
                            "Fourier-SVD",
                        )
                        .on_hover_text("Fast FFT-guided SVD destriping")
                        .changed()
                    {
                        changed = true;
                    }
                });
        });

        changed
    }

    /// Show Fourier-SVD specific parameters (NO advanced section)
    fn show_fourier_svd_params(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        ui.label("Fourier-SVD Parameters")
            .on_hover_text("Fast FFT-guided SVD destriping algorithm");

        ui.horizontal(|ui| {
            ui.label("FFT Alpha:").on_hover_text(
                "FFT Trust Factor (0.0 - 5.0).\n\
                 Higher = more aggressive streak detection.\n\
                 1.0 = standard.",
            );
            if ui
                .add(egui::Slider::new(&mut self.fft_alpha, 0.0..=5.0).step_by(0.1))
                .changed()
            {
                changed = true;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Notch Width:").on_hover_text(
                "Selectivity of vertical frequency notch filter.\n\
                 Larger = accepts more off-axis frequencies.",
            );
            if ui
                .add(egui::Slider::new(&mut self.notch_width, 0.5..=5.0).step_by(0.1))
                .changed()
            {
                changed = true;
            }
        });

        // Processing axis (needed for volume processing)
        ui.add_space(4.0);
        changed |= self.show_processing_axis(ui);

        changed
    }

    /// Show MultiscaleStreak parameters: sigma + scales in Tier 1, BM3D in Advanced
    fn show_multiscale_streak_params(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        // Tier 1: Sigma and Scales
        changed |= self.show_sigma_control(ui);

        ui.horizontal(|ui| {
            ui.label("Scales:").on_hover_text(
                "Number of pyramid scales for multi-scale processing.\n\
                 0 = Auto (recommended): automatically determines based on image width.\n\
                 Higher values handle wider streaks but are slower.",
            );
            if ui
                .add(egui::DragValue::new(&mut self.num_scales).range(0..=6))
                .changed()
            {
                changed = true;
            }
            if self.num_scales == 0 {
                ui.label("(Auto)");
            }
        });

        ui.add_space(4.0);

        // Advanced section with BM3D params
        changed |= self.show_advanced_bm3d_section(ui);

        changed
    }

    /// Show BM3D parameters for Streak and Generic modes
    fn show_bm3d_params(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        // Tier 1: Sigma only
        changed |= self.show_sigma_control(ui);

        ui.add_space(4.0);

        // Advanced section with BM3D params
        changed |= self.show_advanced_bm3d_section(ui);

        changed
    }

    /// Show sigma control with auto checkbox
    fn show_sigma_control(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        ui.horizontal(|ui| {
            ui.label("Sigma:").on_hover_text(
                "Noise level estimate (sigma_random).\n\
                 Higher values = stronger denoising.\n\
                 Typical range: 0.001 - 0.5\n\
                 Supports scientific notation (e.g., 5e-3)",
            );

            // Auto sigma checkbox
            if ui
                .checkbox(&mut self.auto_sigma, "Auto")
                .on_hover_text("Automatically estimate noise level from image data")
                .changed()
            {
                changed = true;
            }

            // Disable drag value if auto is selected
            ui.add_enabled_ui(!self.auto_sigma, |ui| {
                let sigma_response = ui.add(
                    egui::DragValue::new(&mut self.sigma_random)
                        .speed(0.0001)
                        .range(0.0..=0.5)
                        .max_decimals(4),
                );
                if sigma_response.changed() {
                    changed = true;
                }
            });
        });

        changed
    }

    /// Show processing axis selector
    fn show_processing_axis(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        ui.horizontal(|ui| {
            ui.label("Process Axis:").on_hover_text(
                "Which dimension to iterate over for processing.\n\
                 For [angles, Y, X] data:\n\
                   Axis 0: Process [Y, X] slices (unusual)\n\
                   Axis 1: Process [angles, X] sinograms (default)\n\
                   Axis 2: Process [angles, Y] slices (unusual)",
            );

            egui::ComboBox::from_id_salt("processing_axis")
                .selected_text(format!(
                    "Axis {} (D{})",
                    self.processing_axis, self.processing_axis
                ))
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

        changed
    }

    /// Show advanced BM3D parameters section (collapsible)
    fn show_advanced_bm3d_section(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        ui.horizontal(|ui| {
            if ui
                .selectable_label(self.show_advanced, "⚙ Advanced")
                .on_hover_text("Show advanced BM3D block-matching parameters")
                .clicked()
            {
                self.show_advanced = !self.show_advanced;
            }
        });

        if self.show_advanced {
            ui.indent("advanced_params", |ui| {
                // Patch size
                ui.horizontal(|ui| {
                    ui.label("Patch Size:").on_hover_text(
                        "Size of patches for block matching.\n\
                         Larger = smoother but slower.\n\
                         Smaller = preserves fine details.",
                    );

                    egui::ComboBox::from_id_salt("patch_size")
                        .selected_text(format!("{}", self.patch_size))
                        .show_ui(ui, |ui| {
                            for size in [4, 8, 16] {
                                if ui
                                    .selectable_value(
                                        &mut self.patch_size,
                                        size,
                                        format!("{}", size),
                                    )
                                    .changed()
                                {
                                    changed = true;
                                }
                            }
                        });
                });

                // Search window
                ui.horizontal(|ui| {
                    ui.label("Search Window:").on_hover_text(
                        "Size of area to search for similar patches.\n\
                         Larger = better matches but slower.\n\
                         Range: 16-64",
                    );

                    if ui
                        .add(egui::Slider::new(&mut self.search_window, 16..=64).step_by(4.0))
                        .changed()
                    {
                        changed = true;
                    }
                });

                // Max matches
                ui.horizontal(|ui| {
                    ui.label("Max Matches:").on_hover_text(
                        "Maximum similar patches per group.\n\
                         More = better denoising but slower.\n\
                         Range: 8-64",
                    );

                    if ui
                        .add(egui::Slider::new(&mut self.max_matches, 8..=64).step_by(4.0))
                        .changed()
                    {
                        changed = true;
                    }
                });

                // Processing axis
                changed |= self.show_processing_axis(ui);

                // Reset to defaults button
                if ui
                    .button("Reset to Defaults")
                    .on_hover_text("Reset advanced parameters to their default values")
                    .clicked()
                {
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
