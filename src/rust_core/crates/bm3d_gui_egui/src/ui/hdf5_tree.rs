use crate::data::Hdf5Entry;
use eframe::egui;
use std::collections::HashSet;

/// Selection result from the HDF5 tree browser.
#[derive(Debug, Clone)]
pub struct Hdf5TreeSelection {
    pub dataset_path: String,
    pub shape: Vec<usize>,
}

/// HDF5 tree browser widget.
pub struct Hdf5TreeBrowser {
    entries: Vec<Hdf5Entry>,
    expanded: HashSet<String>,
    selected_path: Option<String>,
}

impl Hdf5TreeBrowser {
    pub fn new(entries: Vec<Hdf5Entry>) -> Self {
        // Auto-expand root level
        let mut expanded = HashSet::new();
        for entry in &entries {
            if let Hdf5Entry::Group { path, .. } = entry {
                expanded.insert(path.clone());
            }
        }
        Self {
            entries,
            expanded,
            selected_path: None,
        }
    }

    /// Show the tree browser and return selection if a 3D dataset is selected.
    pub fn show(&mut self, ui: &mut egui::Ui) -> Option<Hdf5TreeSelection> {
        let mut selection = None;

        egui::ScrollArea::vertical()
            .max_height(200.0)
            .show(ui, |ui| {
                for entry in self.entries.clone() {
                    if let Some(sel) = self.show_entry(ui, &entry, 0) {
                        selection = Some(sel);
                    }
                }
            });

        selection
    }

    fn show_entry(
        &mut self,
        ui: &mut egui::Ui,
        entry: &Hdf5Entry,
        depth: usize,
    ) -> Option<Hdf5TreeSelection> {
        let mut selection = None;
        let indent = depth as f32 * 16.0;

        ui.horizontal(|ui| {
            ui.add_space(indent);

            match entry {
                Hdf5Entry::Group {
                    name,
                    path,
                    children,
                } => {
                    let is_expanded = self.expanded.contains(path);
                    let icon = if is_expanded { "▼" } else { "▶" };

                    if ui.small_button(icon)
                        .on_hover_text(if is_expanded { "Collapse group" } else { "Expand group" })
                        .clicked() {
                        if is_expanded {
                            self.expanded.remove(path);
                        } else {
                            self.expanded.insert(path.clone());
                        }
                    }

                    ui.label(format!("📁 {}", name));
                }
                Hdf5Entry::Dataset {
                    name,
                    path,
                    shape,
                    dtype,
                } => {
                    ui.add_space(20.0); // Align with folder icons

                    let is_3d = shape.len() == 3;
                    let is_selected = self.selected_path.as_ref() == Some(path);

                    let label = if is_3d {
                        format!("📊 {} [{:?}]", name, shape)
                    } else {
                        format!("📄 {} [{:?}] ({}D)", name, shape, shape.len())
                    };

                    let tooltip = if is_3d {
                        "Click to select this 3D dataset for loading"
                    } else {
                        "Only 3D datasets can be loaded"
                    };
                    let response = ui.selectable_label(is_selected, label)
                        .on_hover_text(tooltip);

                    if response.clicked() && is_3d {
                        self.selected_path = Some(path.clone());
                        selection = Some(Hdf5TreeSelection {
                            dataset_path: path.clone(),
                            shape: shape.clone(),
                        });
                    }

                    if !is_3d {
                        ui.label(egui::RichText::new("(not 3D)").small().weak());
                    }
                }
            }
        });

        // Show children if expanded
        if let Hdf5Entry::Group { path, children, .. } = entry {
            if self.expanded.contains(path) {
                for child in children {
                    if let Some(sel) = self.show_entry(ui, child, depth + 1) {
                        selection = Some(sel);
                    }
                }
            }
        }

        selection
    }

    pub fn selected_path(&self) -> Option<&String> {
        self.selected_path.as_ref()
    }
}
