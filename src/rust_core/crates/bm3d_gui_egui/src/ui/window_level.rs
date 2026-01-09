use eframe::egui;

/// Window/Level (contrast/brightness) controls.
/// Controls the display range for image visualization.
pub struct WindowLevel {
    /// Minimum display value (values below are clipped to black)
    pub min: f32,
    /// Maximum display value (values above are clipped to white)
    pub max: f32,
    /// Data range for slider bounds
    data_min: f32,
    data_max: f32,
    /// Whether auto mode is active
    auto_mode: bool,
}

impl Default for WindowLevel {
    fn default() -> Self {
        Self {
            min: 0.0,
            max: 1.0,
            data_min: 0.0,
            data_max: 1.0,
            auto_mode: true,
        }
    }
}

impl WindowLevel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update the data range (call when loading new data or changing slice).
    pub fn set_data_range(&mut self, min: f32, max: f32) {
        self.data_min = min;
        self.data_max = max;
        if self.auto_mode {
            self.min = min;
            self.max = max;
        }
    }

    /// Apply auto mode - set display range to data range.
    pub fn auto(&mut self) {
        self.min = self.data_min;
        self.max = self.data_max;
        self.auto_mode = true;
    }

    /// Map a raw value to normalized [0, 1] using current window/level.
    pub fn normalize(&self, value: f32) -> f32 {
        if (self.max - self.min).abs() < f32::EPSILON {
            return 0.5;
        }
        ((value - self.min) / (self.max - self.min)).clamp(0.0, 1.0)
    }

    /// Show window/level controls. Returns true if settings changed.
    pub fn show(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        ui.horizontal(|ui| {
            ui.label("Window/Level:");

            // Min slider
            ui.label("Min:");
            let min_response = ui.add(
                egui::DragValue::new(&mut self.min)
                    .speed(0.01 * (self.data_max - self.data_min).abs().max(1.0))
                    .range(self.data_min..=self.max)
                    .max_decimals(2),
            );
            if min_response.changed() {
                self.auto_mode = false;
                changed = true;
            }

            ui.separator();

            // Max slider
            ui.label("Max:");
            let max_response = ui.add(
                egui::DragValue::new(&mut self.max)
                    .speed(0.01 * (self.data_max - self.data_min).abs().max(1.0))
                    .range(self.min..=self.data_max)
                    .max_decimals(2),
            );
            if max_response.changed() {
                self.auto_mode = false;
                changed = true;
            }

            ui.separator();

            // Auto button
            let auto_text = if self.auto_mode { "Auto ✓" } else { "Auto" };
            if ui.button(auto_text).clicked() {
                self.auto();
                changed = true;
            }
        });

        // Show current range info
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(format!(
                    "Data range: [{:.2}, {:.2}]  |  Display: [{:.2}, {:.2}]",
                    self.data_min, self.data_max, self.min, self.max
                ))
                .small()
                .weak(),
            );
        });

        changed
    }
}
