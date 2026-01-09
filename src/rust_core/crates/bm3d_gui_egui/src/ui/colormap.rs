use eframe::egui;

/// Available colormaps for visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Colormap {
    #[default]
    Grayscale,
    Viridis,
    Inferno,
    Plasma,
}

/// RdBu (Red-Blue) diverging colormap for difference visualization.
/// Maps: -1 -> Blue, 0 -> White, +1 -> Red
pub struct DivergingColormap;

impl DivergingColormap {
    /// Map a value in range [-1, 1] to RGB color using RdBu colormap.
    /// Negative values -> Blue, Zero -> White, Positive values -> Red
    pub fn rdbu(t: f32) -> [u8; 3] {
        let t = t.clamp(-1.0, 1.0);

        if t < 0.0 {
            // Blue side: interpolate from Blue to White
            let s = 1.0 + t; // s goes from 0 (t=-1) to 1 (t=0)
            let r = (33.0 + s * (255.0 - 33.0)) as u8;   // 33 -> 255
            let g = (102.0 + s * (255.0 - 102.0)) as u8; // 102 -> 255
            let b = (172.0 + s * (255.0 - 172.0)) as u8; // 172 -> 255
            [r, g, b]
        } else {
            // Red side: interpolate from White to Red
            let s = t; // s goes from 0 (t=0) to 1 (t=1)
            let r = 255;                                    // 255 -> 255
            let g = (255.0 - s * (255.0 - 58.0)) as u8;    // 255 -> 58
            let b = (255.0 - s * (255.0 - 56.0)) as u8;    // 255 -> 56
            [r, g, b]
        }
    }

    /// Generate a 256-entry lookup table for RdBu colormap.
    /// Index 0 = -1 (blue), Index 127/128 = 0 (white), Index 255 = +1 (red)
    pub fn generate_lut() -> [[u8; 3]; 256] {
        let mut lut = [[0u8; 3]; 256];
        for (i, entry) in lut.iter_mut().enumerate() {
            // Map [0, 255] to [-1, 1]
            let t = (i as f32 / 255.0) * 2.0 - 1.0;
            *entry = Self::rdbu(t);
        }
        lut
    }
}

impl Colormap {
    pub const ALL: [Colormap; 4] = [
        Colormap::Grayscale,
        Colormap::Viridis,
        Colormap::Inferno,
        Colormap::Plasma,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            Colormap::Grayscale => "Grayscale",
            Colormap::Viridis => "Viridis",
            Colormap::Inferno => "Inferno",
            Colormap::Plasma => "Plasma",
        }
    }

    /// Map a normalized value [0, 1] to RGB color.
    pub fn map(&self, t: f32) -> [u8; 3] {
        let t = t.clamp(0.0, 1.0);
        match self {
            Colormap::Grayscale => {
                let v = (t * 255.0) as u8;
                [v, v, v]
            }
            Colormap::Viridis => Self::viridis(t),
            Colormap::Inferno => Self::inferno(t),
            Colormap::Plasma => Self::plasma(t),
        }
    }

    /// Generate a 256-entry lookup table for fast colormap application.
    pub fn generate_lut(&self) -> [[u8; 3]; 256] {
        let mut lut = [[0u8; 3]; 256];
        for (i, entry) in lut.iter_mut().enumerate() {
            *entry = self.map(i as f32 / 255.0);
        }
        lut
    }

    // Viridis colormap (perceptually uniform, colorblind-friendly)
    fn viridis(t: f32) -> [u8; 3] {
        // Simplified Viridis approximation using key points
        let (r, g, b) = if t < 0.25 {
            let s = t / 0.25;
            (
                68.0 + s * (49.0 - 68.0),
                1.0 + s * (54.0 - 1.0),
                84.0 + s * (149.0 - 84.0),
            )
        } else if t < 0.5 {
            let s = (t - 0.25) / 0.25;
            (
                49.0 + s * (53.0 - 49.0),
                54.0 + s * (183.0 - 54.0),
                149.0 + s * (121.0 - 149.0),
            )
        } else if t < 0.75 {
            let s = (t - 0.5) / 0.25;
            (
                53.0 + s * (180.0 - 53.0),
                183.0 + s * (222.0 - 183.0),
                121.0 + s * (44.0 - 121.0),
            )
        } else {
            let s = (t - 0.75) / 0.25;
            (
                180.0 + s * (253.0 - 180.0),
                222.0 + s * (231.0 - 222.0),
                44.0 + s * (37.0 - 44.0),
            )
        };
        [r as u8, g as u8, b as u8]
    }

    // Inferno colormap (perceptually uniform, black to yellow through red)
    fn inferno(t: f32) -> [u8; 3] {
        let (r, g, b) = if t < 0.25 {
            let s = t / 0.25;
            (
                0.0 + s * 87.0,
                0.0 + s * 16.0,
                4.0 + s * (110.0 - 4.0),
            )
        } else if t < 0.5 {
            let s = (t - 0.25) / 0.25;
            (
                87.0 + s * (188.0 - 87.0),
                16.0 + s * (55.0 - 16.0),
                110.0 + s * (84.0 - 110.0),
            )
        } else if t < 0.75 {
            let s = (t - 0.5) / 0.25;
            (
                188.0 + s * (249.0 - 188.0),
                55.0 + s * (142.0 - 55.0),
                84.0 + s * (9.0 - 84.0),
            )
        } else {
            let s = (t - 0.75) / 0.25;
            (
                249.0 + s * (252.0 - 249.0),
                142.0 + s * (255.0 - 142.0),
                9.0 + s * (164.0 - 9.0),
            )
        };
        [r as u8, g as u8, b as u8]
    }

    // Plasma colormap (perceptually uniform, blue to yellow through magenta)
    fn plasma(t: f32) -> [u8; 3] {
        let (r, g, b) = if t < 0.25 {
            let s = t / 0.25;
            (
                13.0 + s * (126.0 - 13.0),
                8.0 + s * (3.0 - 8.0),
                135.0 + s * (168.0 - 135.0),
            )
        } else if t < 0.5 {
            let s = (t - 0.25) / 0.25;
            (
                126.0 + s * (204.0 - 126.0),
                3.0 + s * (71.0 - 3.0),
                168.0 + s * (120.0 - 168.0),
            )
        } else if t < 0.75 {
            let s = (t - 0.5) / 0.25;
            (
                204.0 + s * (248.0 - 204.0),
                71.0 + s * (149.0 - 71.0),
                120.0 + s * (64.0 - 120.0),
            )
        } else {
            let s = (t - 0.75) / 0.25;
            (
                248.0 + s * (240.0 - 248.0),
                149.0 + s * (249.0 - 149.0),
                64.0 + s * (33.0 - 64.0),
            )
        };
        [r as u8, g as u8, b as u8]
    }
}

/// Widget for selecting colormap.
pub struct ColormapSelector {
    current: Colormap,
}

impl Default for ColormapSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl ColormapSelector {
    pub fn new() -> Self {
        Self {
            current: Colormap::default(),
        }
    }

    pub fn current(&self) -> Colormap {
        self.current
    }

    /// Show colormap selector. Returns true if selection changed.
    pub fn show(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        ui.horizontal(|ui| {
            ui.label("Colormap:");
            egui::ComboBox::from_id_salt("colormap_selector")
                .selected_text(self.current.name())
                .show_ui(ui, |ui| {
                    for cmap in Colormap::ALL {
                        if ui
                            .selectable_value(&mut self.current, cmap, cmap.name())
                            .changed()
                        {
                            changed = true;
                        }
                    }
                });
        });

        changed
    }
}
