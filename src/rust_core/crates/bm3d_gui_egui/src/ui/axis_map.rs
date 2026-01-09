use crate::data::AxisMapping;
use eframe::egui;

/// Widget for configuring axis mapping with H5Web-style auto-swap.
/// When changing one axis to a value used by another, they automatically swap.
pub struct AxisMappingWidget {
    mapping: AxisMapping,
    shape: [usize; 3],
}

impl AxisMappingWidget {
    pub fn new(shape: [usize; 3]) -> Self {
        Self {
            mapping: AxisMapping::default(),
            shape,
        }
    }

    pub fn mapping(&self) -> AxisMapping {
        self.mapping
    }

    /// Show the axis mapping controls. Returns true if mapping changed.
    pub fn show(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let labels = AxisMapping::dimension_labels();
        let shape = self.shape;

        // Store old values for swap detection
        let old_x = self.mapping.x_axis;
        let old_y = self.mapping.y_axis;
        let old_slice = self.mapping.slice_axis;

        ui.horizontal(|ui| {
            ui.label("Axis Mapping:");

            // X axis dropdown
            ui.label("X:");
            let mut new_x = self.mapping.x_axis;
            let x_changed = Self::axis_dropdown_static(ui, "x_axis", &mut new_x, &labels, &shape);
            if x_changed && new_x != old_x {
                self.apply_axis_change(AxisRole::X, new_x, old_x);
                changed = true;
            }

            ui.separator();

            // Y axis dropdown
            ui.label("Y:");
            let mut new_y = self.mapping.y_axis;
            let y_changed = Self::axis_dropdown_static(ui, "y_axis", &mut new_y, &labels, &shape);
            if y_changed && new_y != old_y {
                self.apply_axis_change(AxisRole::Y, new_y, old_y);
                changed = true;
            }

            ui.separator();

            // Slice axis dropdown
            ui.label("Slice:");
            let mut new_slice = self.mapping.slice_axis;
            let slice_changed =
                Self::axis_dropdown_static(ui, "slice_axis", &mut new_slice, &labels, &shape);
            if slice_changed && new_slice != old_slice {
                self.apply_axis_change(AxisRole::Slice, new_slice, old_slice);
                changed = true;
            }

            ui.separator();

            if ui.button("Reset to Standard").clicked() {
                self.mapping = AxisMapping::default();
                changed = true;
            }
        });

        // Show current dimension sizes and mapping
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(format!(
                    "Shape: D0={}, D1={}, D2={}  |  Display: {}×{}, {} slices",
                    shape[0],
                    shape[1],
                    shape[2],
                    shape[self.mapping.x_axis],
                    shape[self.mapping.y_axis],
                    shape[self.mapping.slice_axis],
                ))
                .small()
                .weak(),
            );
        });

        changed
    }

    /// Apply an axis change with automatic swap to maintain valid mapping.
    /// H5Web style: when X takes Y's value, Y gets X's old value (swap).
    fn apply_axis_change(&mut self, changed_role: AxisRole, new_value: usize, old_value: usize) {
        // Find which other axis has the new_value and swap it
        match changed_role {
            AxisRole::X => {
                if self.mapping.y_axis == new_value {
                    self.mapping.y_axis = old_value;
                } else if self.mapping.slice_axis == new_value {
                    self.mapping.slice_axis = old_value;
                }
                self.mapping.x_axis = new_value;
            }
            AxisRole::Y => {
                if self.mapping.x_axis == new_value {
                    self.mapping.x_axis = old_value;
                } else if self.mapping.slice_axis == new_value {
                    self.mapping.slice_axis = old_value;
                }
                self.mapping.y_axis = new_value;
            }
            AxisRole::Slice => {
                if self.mapping.x_axis == new_value {
                    self.mapping.x_axis = old_value;
                } else if self.mapping.y_axis == new_value {
                    self.mapping.y_axis = old_value;
                }
                self.mapping.slice_axis = new_value;
            }
        }
    }

    fn axis_dropdown_static(
        ui: &mut egui::Ui,
        id: &str,
        current: &mut usize,
        labels: &[&str; 3],
        shape: &[usize; 3],
    ) -> bool {
        let mut changed = false;

        egui::ComboBox::from_id_salt(id)
            .selected_text(format!("{} ({})", labels[*current], shape[*current]))
            .width(80.0)
            .show_ui(ui, |ui| {
                for (idx, label) in labels.iter().enumerate() {
                    let text = format!("{} ({})", label, shape[idx]);
                    if ui.selectable_value(current, idx, text).changed() {
                        changed = true;
                    }
                }
            });

        changed
    }
}

#[derive(Clone, Copy)]
enum AxisRole {
    X,
    Y,
    Slice,
}
