use bm3d_core::{Bm3dConfig, MultiscaleConfig, RingRemovalMode};

use super::method_info::show_method_info;
use eframe::egui;

/// BM3D processing parameters with method-specific UI panels.
pub struct Bm3dParameters {
    pub mode: RingRemovalMode,

    /// Random noise standard deviation.
    pub sigma_random: f32,
    pub auto_sigma: bool,

    pub patch_size: usize,
    pub search_window: usize,
    pub max_matches: usize,
    /// Speed-first transform mode (8x8 Hadamard); can increase block artifacts.
    pub use_hadamard_fast_mode: bool,
    /// Which axis to iterate over for sinogram processing (0, 1, or 2).
    /// Default is 1 (Y axis) for standard tomography data [angles, Y, X].
    pub processing_axis: usize,

    pub fft_alpha: f32,
    pub notch_width: f32,

    pub num_scales: usize, // 0 = auto
    /// The loaded data is already log-transformed (attenuation), so the
    /// multiscale pipeline must not apply its own logarithm.
    pub log_domain_input: bool,

    show_advanced: bool,
}

impl Default for Bm3dParameters {
    fn default() -> Self {
        let bm3d_default = Bm3dConfig::<f32>::default();
        let multiscale_default = MultiscaleConfig::<f32>::default();

        Self {
            mode: RingRemovalMode::default(),
            sigma_random: bm3d_default.sigma_random,
            auto_sigma: true,
            patch_size: bm3d_default.patch_size,
            search_window: bm3d_default.search_window,
            max_matches: bm3d_default.max_matches,
            use_hadamard_fast_mode: false,
            processing_axis: 1,
            fft_alpha: bm3d_default.fft_alpha,
            notch_width: bm3d_default.notch_width,
            num_scales: multiscale_default.num_scales.unwrap_or(0),
            log_domain_input: multiscale_default.log_domain_input,
            show_advanced: false,
        }
    }
}

impl Bm3dParameters {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn to_config(&self) -> Bm3dConfig<f32> {
        let default = Bm3dConfig::<f32>::default();
        let sigma = if self.auto_sigma {
            0.0
        } else {
            self.sigma_random
        };

        Bm3dConfig {
            sigma_random: sigma,
            patch_size: self.patch_size,
            step_size: self.patch_size / 2,
            search_window: self.search_window,
            max_matches: self.max_matches,
            fft_alpha: self.fft_alpha,
            notch_width: self.notch_width,
            use_hadamard_fast_path: Some(self.use_hadamard_fast_mode),
            ..default
        }
    }

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
            log_domain_input: self.log_domain_input,
            ..default
        }
    }

    pub fn uses_multiscale(&self) -> bool {
        self.mode == RingRemovalMode::MultiscaleStreak
    }

    /// Show parameter controls. Returns true if any parameter changed.
    pub fn show(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        ui.heading("Processing Parameters");

        changed |= self.show_mode_selection(ui);

        show_method_info(ui, self.mode);

        ui.add_space(4.0);
        ui.separator();
        ui.add_space(4.0);

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

    fn show_mode_selection(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        ui.horizontal(|ui| {
            ui.label("Method:").on_hover_text(
                "Streak: Default single-scale BM3D for ring artifacts\n\
                 Multiscale Streak: Best quality for wide ring artifacts\n\
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

        ui.add_space(4.0);
        changed |= self.show_processing_axis(ui);

        changed
    }

    fn show_multiscale_streak_params(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

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

        if ui
            .checkbox(
                &mut self.log_domain_input,
                "Input is already log-transformed (attenuation)",
            )
            .on_hover_text(
                "Check this for attenuation sinograms (-ln of transmission), the usual\n\
                 normalized product. Leave unchecked for linear transmission data in\n\
                 [0, 1], which the pipeline log-transforms itself.\n\
                 Wrong setting = a second logarithm and unusable output.",
            )
            .changed()
        {
            changed = true;
        }

        ui.add_space(4.0);

        changed |= self.show_advanced_bm3d_section(ui);

        changed
    }

    fn show_bm3d_params(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        changed |= self.show_sigma_control(ui);

        ui.add_space(4.0);

        changed |= self.show_advanced_bm3d_section(ui);

        changed
    }

    fn show_sigma_control(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        ui.horizontal(|ui| {
            ui.label("Sigma:").on_hover_text(
                "Noise level estimate (sigma_random).\n\
                 Higher values = stronger denoising.\n\
                 Typical range: 0.001 - 0.5\n\
                 Supports scientific notation (e.g., 5e-3)",
            );

            if ui
                .checkbox(&mut self.auto_sigma, "Auto")
                .on_hover_text("Automatically estimate noise level from image data")
                .changed()
            {
                changed = true;
            }

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
                ui.horizontal(|ui| {
                    ui.label("Patch Size:").on_hover_text(
                        "Size of patches for block matching.\n\
                         Larger = smoother but slower.\n\
                         Smaller = preserves fine details.\n\
                         Any size works; 7 or 8 recommended.\n\
                         The 8x8 fast mode only applies at patch size 8.",
                    );

                    if ui
                        .add(egui::Slider::new(&mut self.patch_size, 4..=16))
                        .changed()
                    {
                        changed = true;
                    }
                });

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

                ui.horizontal(|ui| {
                    let response = ui
                        .checkbox(&mut self.use_hadamard_fast_mode, "Fast Mode (Hadamard 8x8)")
                        .on_hover_text(
                            "Speed-first transform for 8x8 patches.\n\
                         May introduce blocky/patchy artifacts near strong edges.\n\
                         Keep OFF for highest quality.",
                        );
                    if response.changed() {
                        changed = true;
                    }
                });

                changed |= self.show_processing_axis(ui);

                if ui
                    .button("Reset to Defaults")
                    .on_hover_text("Reset advanced parameters to their default values")
                    .clicked()
                {
                    let bm3d_default = Bm3dConfig::<f32>::default();
                    self.patch_size = bm3d_default.patch_size;
                    self.search_window = bm3d_default.search_window;
                    self.max_matches = bm3d_default.max_matches;
                    self.use_hadamard_fast_mode = false;
                    self.processing_axis = 1;
                    changed = true;
                }
            });
        }

        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_patch_size_maps_to_config() {
        let mut params = Bm3dParameters::new();
        params.patch_size = 7;
        let config = params.to_config();
        assert_eq!(config.patch_size, 7);
        assert_eq!(config.step_size, 3);
    }

    /// The GUI has no search-stride control, so it leaves `search_step` to
    /// the per-mode default of bm3d_core.
    #[test]
    fn search_step_is_left_to_core_default() {
        let mut params = Bm3dParameters::new();
        for patch_size in [params.patch_size, 7] {
            params.patch_size = patch_size;
            assert_eq!(params.to_config().search_step, None);
            assert_eq!(params.to_multiscale_config().bm3d_config.search_step, None);
        }
    }

    #[test]
    fn default_patch_size_matches_core_default() {
        let params = Bm3dParameters::new();
        assert_eq!(params.patch_size, Bm3dConfig::<f32>::default().patch_size);
    }

    #[test]
    fn log_domain_toggle_maps_to_multiscale_config() {
        let mut params = Bm3dParameters::new();
        assert_eq!(
            params.log_domain_input,
            MultiscaleConfig::<f32>::default().log_domain_input
        );
        params.log_domain_input = true;
        assert!(params.to_multiscale_config().log_domain_input);
        params.log_domain_input = false;
        assert!(!params.to_multiscale_config().log_domain_input);
    }
}
