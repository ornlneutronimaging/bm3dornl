use eframe::egui;
use ndarray::Array3;
use std::path::PathBuf;

const TIFF_GRAY32_FLOAT_BYTES_PER_PIXEL: u64 = std::mem::size_of::<f32>() as u64;
const CLASSIC_TIFF_SAFETY_MARGIN_BYTES: u64 = 64 * 1024 * 1024;
const CLASSIC_TIFF_MAX_SAFE_BYTES: u64 = u32::MAX as u64 - CLASSIC_TIFF_SAFETY_MARGIN_BYTES;

/// What data to save
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SaveDataType {
    #[default]
    Original,
    Processed,
    Difference,
}

impl SaveDataType {
    pub fn name(&self) -> &'static str {
        match self {
            SaveDataType::Original => "Original",
            SaveDataType::Processed => "Processed",
            SaveDataType::Difference => "Difference",
        }
    }
}

/// Output format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SaveFormat {
    #[default]
    Tiff,
    Hdf5,
}

impl SaveFormat {
    pub fn name(&self) -> &'static str {
        match self {
            SaveFormat::Tiff => "TIFF Stack",
            SaveFormat::Hdf5 => "HDF5",
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            SaveFormat::Tiff => "tiff",
            SaveFormat::Hdf5 => "h5",
        }
    }
}

/// Save progress state
#[derive(Debug, Clone, Default)]
pub enum SaveState {
    #[default]
    Idle,
    Saving {
        progress: f32,
        message: String,
    },
    Completed {
        path: PathBuf,
    },
    Error(String),
}

/// Save dialog state and UI
pub struct SaveDialog {
    pub is_open: bool,
    pub data_type: SaveDataType,
    pub format: SaveFormat,
    pub hdf5_dataset_path: String,
    pub state: SaveState,
}

impl Default for SaveDialog {
    fn default() -> Self {
        Self::new()
    }
}

impl SaveDialog {
    pub fn new() -> Self {
        Self {
            is_open: false,
            data_type: SaveDataType::default(),
            format: SaveFormat::default(),
            hdf5_dataset_path: "/data".to_string(),
            state: SaveState::Idle,
        }
    }

    pub fn open(&mut self) {
        self.is_open = true;
        self.state = SaveState::Idle;
    }

    pub fn close(&mut self) {
        self.is_open = false;
    }

    /// Show save dialog. Returns Some(SaveRequest) if user confirms save.
    pub fn show(&mut self, ctx: &egui::Context, has_processed: bool) -> Option<SaveRequest> {
        if !self.is_open {
            return None;
        }

        let mut save_request = None;
        let mut should_close = false;

        egui::Window::new("Save Volume")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.set_min_width(300.0);

                match &self.state {
                    SaveState::Saving { progress, message } => {
                        ui.label(message);
                        ui.add(egui::ProgressBar::new(*progress).show_percentage());
                    }
                    SaveState::Completed { path } => {
                        ui.colored_label(
                            egui::Color32::GREEN,
                            format!("✓ Saved to: {}", path.display()),
                        );
                        ui.add_space(10.0);
                        if ui.button("Close").clicked() {
                            should_close = true;
                        }
                    }
                    SaveState::Error(msg) => {
                        ui.colored_label(egui::Color32::RED, format!("✗ Error: {}", msg));
                        ui.add_space(10.0);
                        if ui.button("Close").clicked() {
                            should_close = true;
                        }
                    }
                    SaveState::Idle => {
                        // Data type selection
                        ui.horizontal(|ui| {
                            ui.label("Data:")
                                .on_hover_text("Select which data to save");
                            egui::ComboBox::from_id_salt("save_data_type")
                                .selected_text(self.data_type.name())
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut self.data_type,
                                        SaveDataType::Original,
                                        "Original",
                                    ).on_hover_text("Save the original input data");
                                    ui.add_enabled_ui(has_processed, |ui| {
                                        ui.selectable_value(
                                            &mut self.data_type,
                                            SaveDataType::Processed,
                                            "Processed",
                                        ).on_hover_text("Save the BM3D-processed data");
                                        ui.selectable_value(
                                            &mut self.data_type,
                                            SaveDataType::Difference,
                                            "Difference (Original - Processed)",
                                        ).on_hover_text("Save the difference between original and processed");
                                    });
                                });
                        });

                        // Reset to Original if no processed data
                        if !has_processed
                            && (self.data_type == SaveDataType::Processed
                                || self.data_type == SaveDataType::Difference)
                        {
                            self.data_type = SaveDataType::Original;
                        }

                        ui.add_space(5.0);

                        // Format selection
                        ui.horizontal(|ui| {
                            ui.label("Format:")
                                .on_hover_text("Select output file format");
                            egui::ComboBox::from_id_salt("save_format")
                                .selected_text(self.format.name())
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut self.format, SaveFormat::Tiff, "TIFF Stack")
                                        .on_hover_text("Multi-page TIFF with 32-bit float values; large stacks are saved as BigTIFF automatically");
                                    ui.selectable_value(&mut self.format, SaveFormat::Hdf5, "HDF5")
                                        .on_hover_text("HDF5 file with single dataset");
                                });
                        });

                        // HDF5-specific options
                        if self.format == SaveFormat::Hdf5 {
                            ui.add_space(5.0);
                            ui.horizontal(|ui| {
                                ui.label("Dataset path:")
                                    .on_hover_text("Path within HDF5 file where data will be stored (e.g., /data or /entry/data)");
                                ui.text_edit_singleline(&mut self.hdf5_dataset_path);
                            });
                        }

                        ui.add_space(15.0);

                        // Action buttons
                        ui.horizontal(|ui| {
                            if ui.button("Cancel").clicked() {
                                should_close = true;
                            }

                            ui.add_space(10.0);

                            if ui.button("Save...")
                                .on_hover_text("Choose file location and save")
                                .clicked() {
                                // Open file dialog
                                let filter_name = self.format.name();
                                let filter_ext = self.format.extension();

                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter(filter_name, &[filter_ext])
                                    .set_file_name(format!("volume.{}", filter_ext))
                                    .save_file()
                                {
                                    save_request = Some(SaveRequest {
                                        path,
                                        data_type: self.data_type,
                                        format: self.format,
                                        hdf5_dataset_path: self.hdf5_dataset_path.clone(),
                                    });
                                }
                            }
                        });
                    }
                }
            });

        if should_close {
            self.close();
        }

        save_request
    }
}

/// Request to save data
#[derive(Debug, Clone)]
pub struct SaveRequest {
    pub path: PathBuf,
    pub data_type: SaveDataType,
    pub format: SaveFormat,
    pub hdf5_dataset_path: String,
}

/// Save volume data to file
pub fn save_volume(data: &Array3<f32>, request: &SaveRequest) -> Result<(), String> {
    match request.format {
        SaveFormat::Tiff => save_as_tiff(data, &request.path),
        SaveFormat::Hdf5 => save_as_hdf5(data, &request.path, &request.hdf5_dataset_path),
    }
}

fn save_as_tiff(data: &Array3<f32>, path: &PathBuf) -> Result<(), String> {
    if should_use_bigtiff(data.dim()) {
        save_as_bigtiff(data, path)
    } else {
        save_as_classic_tiff(data, path)
    }
}

fn estimated_tiff_payload_bytes((num_slices, height, width): (usize, usize, usize)) -> Option<u64> {
    let num_slices = u64::try_from(num_slices).ok()?;
    let height = u64::try_from(height).ok()?;
    let width = u64::try_from(width).ok()?;

    num_slices
        .checked_mul(height)?
        .checked_mul(width)?
        .checked_mul(TIFF_GRAY32_FLOAT_BYTES_PER_PIXEL)
}

fn should_use_bigtiff(dim: (usize, usize, usize)) -> bool {
    match estimated_tiff_payload_bytes(dim) {
        Some(bytes) => bytes > CLASSIC_TIFF_MAX_SAFE_BYTES,
        None => true,
    }
}

fn save_as_classic_tiff(data: &Array3<f32>, path: &PathBuf) -> Result<(), String> {
    use std::fs::File;
    use std::io::BufWriter;
    use tiff::encoder::TiffEncoder;

    let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
    let writer = BufWriter::new(file);

    let mut encoder =
        TiffEncoder::new(writer).map_err(|e| format!("Failed to create TIFF encoder: {}", e))?;

    write_tiff_pages(data, &mut encoder)
}

fn save_as_bigtiff(data: &Array3<f32>, path: &PathBuf) -> Result<(), String> {
    use std::fs::File;
    use std::io::BufWriter;
    use tiff::encoder::TiffEncoder;

    let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
    let writer = BufWriter::new(file);

    let mut encoder = TiffEncoder::new_big(writer)
        .map_err(|e| format!("Failed to create BigTIFF encoder: {}", e))?;

    write_tiff_pages(data, &mut encoder)
}

fn write_tiff_pages<W, K>(
    data: &Array3<f32>,
    encoder: &mut tiff::encoder::TiffEncoder<W, K>,
) -> Result<(), String>
where
    W: std::io::Write + std::io::Seek,
    K: tiff::encoder::TiffKind,
{
    use std::borrow::Cow;
    use tiff::encoder::colortype::Gray32Float;

    let (num_slices, height, width) = data.dim();
    let width = u32::try_from(width).map_err(|_| "TIFF width exceeds u32 limit".to_string())?;
    let height = u32::try_from(height).map_err(|_| "TIFF height exceeds u32 limit".to_string())?;

    for slice_idx in 0..num_slices {
        let slice = data.slice(ndarray::s![slice_idx, .., ..]);
        let slice_data: Cow<'_, [f32]> = match slice.as_slice_memory_order() {
            Some(values) => Cow::Borrowed(values),
            None => Cow::Owned(slice.iter().copied().collect()),
        };

        encoder
            .write_image::<Gray32Float>(width, height, &slice_data)
            .map_err(|e| format!("Failed to write TIFF page {}: {}", slice_idx, e))?;
    }

    Ok(())
}

fn save_as_hdf5(data: &Array3<f32>, path: &PathBuf, dataset_path: &str) -> Result<(), String> {
    use hdf5_metno::File as H5File;

    let file = H5File::create(path).map_err(|e| format!("Failed to create HDF5 file: {}", e))?;

    // Create parent groups if needed
    let parts: Vec<&str> = dataset_path.trim_start_matches('/').split('/').collect();
    if parts.len() > 1 {
        let mut current_path = String::new();
        for part in &parts[..parts.len() - 1] {
            current_path.push('/');
            current_path.push_str(part);
            if file.group(&current_path).is_err() {
                file.create_group(&current_path)
                    .map_err(|e| format!("Failed to create group {}: {}", current_path, e))?;
            }
        }
    }

    // Create dataset
    let builder = file.new_dataset_builder();
    builder
        .with_data(data)
        .create(dataset_path)
        .map_err(|e| format!("Failed to create dataset: {}", e))?;

    Ok(())
}

/// Compute difference array (Original - Processed)
pub fn compute_difference(original: &Array3<f32>, processed: &Array3<f32>) -> Array3<f32> {
    original - processed
}
