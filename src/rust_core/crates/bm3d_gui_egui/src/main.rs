mod data;
mod processing;
mod ui;

use data::{build_hdf5_tree, load_hdf5_dataset, load_tiff_stack, Volume3D};
use eframe::egui;
use processing::{ProcessingManager, ProcessingState};
use std::path::PathBuf;
use ui::{
    compute_difference, save_volume, AxisMappingWidget, Bm3dParameters, ColormapSelector,
    CompareView, CompareViewHistogram, Hdf5TreeBrowser, SaveDataType, SaveDialog,
    SingleViewHistogram, SliceViewer, WindowLevel,
};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1400.0, 900.0]),
        ..Default::default()
    };
    eframe::run_native(
        "BM3D Volume Viewer",
        options,
        Box::new(|_cc| Ok(Box::new(App::default()))),
    )
}

/// Which volume to display in single view mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ViewMode {
    #[default]
    Original,
    Processed,
}

/// Display mode: single image or side-by-side comparison
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum DisplayMode {
    #[default]
    Single,
    Compare,
}

struct App {
    // Data state
    volume: Option<Volume3D>,
    processed_volume: Option<Volume3D>,
    file_path: Option<PathBuf>,

    // HDF5 browsing state
    hdf5_tree: Option<Hdf5TreeBrowser>,
    pending_hdf5_path: Option<PathBuf>,

    // UI components
    axis_mapping_widget: Option<AxisMappingWidget>,
    slice_viewer: SliceViewer,
    compare_view: CompareView,
    colormap_selector: ColormapSelector,
    window_level: WindowLevel,

    // Histogram overlays
    single_histogram: SingleViewHistogram,
    compare_histogram: CompareViewHistogram,

    // Processing
    bm3d_params: Bm3dParameters,
    processing_manager: ProcessingManager,
    view_mode: ViewMode,
    display_mode: DisplayMode,

    // Save dialog
    save_dialog: SaveDialog,

    // Display options
    keep_aspect_ratio: bool,

    // Error handling
    error_message: Option<String>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            volume: None,
            processed_volume: None,
            file_path: None,
            hdf5_tree: None,
            pending_hdf5_path: None,
            axis_mapping_widget: None,
            slice_viewer: SliceViewer::default(),
            compare_view: CompareView::default(),
            colormap_selector: ColormapSelector::default(),
            window_level: WindowLevel::default(),
            single_histogram: SingleViewHistogram::new(),
            compare_histogram: CompareViewHistogram::new(),
            bm3d_params: Bm3dParameters::default(),
            processing_manager: ProcessingManager::default(),
            view_mode: ViewMode::default(),
            display_mode: DisplayMode::default(),
            save_dialog: SaveDialog::default(),
            keep_aspect_ratio: true, // Default to maintaining aspect ratio
            error_message: None,
        }
    }
}

impl App {
    fn open_file_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Volume files", &["h5", "hdf5", "nxs", "tif", "tiff"])
            .add_filter("HDF5 files", &["h5", "hdf5", "nxs"])
            .add_filter("TIFF files", &["tif", "tiff"])
            .pick_file()
        {
            self.load_file(path);
        }
    }

    fn load_file(&mut self, path: PathBuf) {
        self.error_message = None;
        self.volume = None;
        self.processed_volume = None;
        self.hdf5_tree = None;
        self.pending_hdf5_path = None;
        self.axis_mapping_widget = None;
        self.slice_viewer.reset();
        self.compare_view.reset();
        self.single_histogram.reset();
        self.compare_histogram.reset();
        self.window_level = WindowLevel::new();
        self.processing_manager.reset();
        self.view_mode = ViewMode::Original;
        self.display_mode = DisplayMode::Single;

        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "h5" | "hdf5" | "nxs" => match build_hdf5_tree(&path) {
                Ok(entries) => {
                    self.hdf5_tree = Some(Hdf5TreeBrowser::new(entries));
                    self.pending_hdf5_path = Some(path.clone());
                    self.file_path = Some(path);
                }
                Err(e) => {
                    self.error_message = Some(format!("Failed to open HDF5: {}", e));
                }
            },
            "tif" | "tiff" => match load_tiff_stack(&path) {
                Ok(vol) => {
                    self.setup_volume(vol);
                    self.file_path = Some(path);
                }
                Err(e) => {
                    self.error_message = Some(format!("Failed to load TIFF: {}", e));
                }
            },
            _ => {
                self.error_message = Some(format!("Unsupported file type: {}", extension));
            }
        }
    }

    fn load_hdf5_dataset(&mut self, dataset_path: &str) {
        if let Some(file_path) = &self.pending_hdf5_path {
            match load_hdf5_dataset(file_path, dataset_path) {
                Ok(vol) => {
                    self.setup_volume(vol);
                    self.hdf5_tree = None;
                }
                Err(e) => {
                    self.error_message = Some(format!("Failed to load dataset: {}", e));
                }
            }
        }
    }

    fn setup_volume(&mut self, volume: Volume3D) {
        let shape = volume.original_shape();
        self.axis_mapping_widget = Some(AxisMappingWidget::new(shape));

        if let Some(slice) = volume.get_slice(0) {
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for &val in slice.iter() {
                if val.is_finite() {
                    min = min.min(val);
                    max = max.max(val);
                }
            }
            self.window_level.set_data_range(min, max);
        }

        self.volume = Some(volume);
        self.processed_volume = None;
        self.slice_viewer.reset();
        self.compare_view.reset();
        self.single_histogram.reset();
        self.compare_histogram.reset();
        self.view_mode = ViewMode::Original;
        self.display_mode = DisplayMode::Single;
    }

    fn update_window_level_for_slice(&mut self) {
        let slice_index = if self.display_mode == DisplayMode::Compare {
            self.compare_view.current_slice()
        } else {
            self.slice_viewer.current_slice()
        };

        let vol = self.current_volume();
        if let Some(vol) = vol {
            if let Some(slice) = vol.get_slice(slice_index) {
                let mut min = f32::INFINITY;
                let mut max = f32::NEG_INFINITY;
                for &val in slice.iter() {
                    if val.is_finite() {
                        min = min.min(val);
                        max = max.max(val);
                    }
                }
                self.window_level.set_data_range(min, max);
            }
        }
    }

    fn current_volume(&self) -> Option<&Volume3D> {
        match self.view_mode {
            ViewMode::Original => self.volume.as_ref(),
            ViewMode::Processed => self.processed_volume.as_ref().or(self.volume.as_ref()),
        }
    }

    fn start_processing(&mut self) {
        if let Some(vol) = &self.volume {
            let raw_data = vol.raw_data().to_owned();
            let mode = self.bm3d_params.mode;
            let config = self.bm3d_params.to_config();

            // Use the processing axis from parameters (configurable in Advanced).
            // Default is axis 1 (Y) for standard tomography data [angles, Y, X].
            // Each slice perpendicular to this axis is processed as a sinogram.
            let processing_axis = self.bm3d_params.processing_axis;

            self.processing_manager
                .start_processing(raw_data, mode, config, processing_axis);
        }
    }

    fn handle_save_request(&mut self, request: &ui::SaveRequest) {
        use ui::save_dialog::SaveState;

        let data = match request.data_type {
            SaveDataType::Original => self.volume.as_ref().map(|v| v.raw_data().to_owned()),
            SaveDataType::Processed => self.processed_volume.as_ref().map(|v| v.raw_data().to_owned()),
            SaveDataType::Difference => {
                if let (Some(orig), Some(proc)) = (&self.volume, &self.processed_volume) {
                    Some(compute_difference(orig.raw_data(), proc.raw_data()))
                } else {
                    None
                }
            }
        };

        if let Some(data) = data {
            self.save_dialog.state = SaveState::Saving {
                progress: 0.5,
                message: "Saving...".to_string(),
            };

            match save_volume(&data, request) {
                Ok(()) => {
                    self.save_dialog.state = SaveState::Completed {
                        path: request.path.clone(),
                    };
                }
                Err(e) => {
                    self.save_dialog.state = SaveState::Error(e);
                }
            }
        } else {
            self.save_dialog.state =
                ui::save_dialog::SaveState::Error("No data available to save".to_string());
        }
    }

    fn show_processing_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Processing");
        ui.separator();

        // Parameters
        self.bm3d_params.show(ui);

        ui.add_space(10.0);

        // Processing controls
        let state = self.processing_manager.state().clone();

        match &state {
            ProcessingState::Idle => {
                let can_process = self.volume.is_some();
                ui.add_enabled_ui(can_process, |ui| {
                    if ui.button("▶ Process")
                        .on_hover_text("Run BM3D denoising with current parameters")
                        .clicked() {
                        self.start_processing();
                    }
                });
            }
            ProcessingState::Processing {
                current_slice,
                total_slices,
            } => {
                let progress = if *total_slices > 0 {
                    *current_slice as f32 / *total_slices as f32
                } else {
                    0.0
                };

                ui.add(
                    egui::ProgressBar::new(progress)
                        .show_percentage()
                        .text(format!("Slice {} / {}", current_slice, total_slices)),
                );

                if ui.button("⏹ Cancel")
                    .on_hover_text("Stop processing")
                    .clicked() {
                    self.processing_manager.cancel();
                }
            }
            ProcessingState::Completed => {
                ui.colored_label(egui::Color32::GREEN, "✓ Processing complete");

                if ui.button("▶ Process Again")
                    .on_hover_text("Re-run BM3D processing with current parameters")
                    .clicked() {
                    self.start_processing();
                }
            }
            ProcessingState::Cancelled => {
                ui.colored_label(egui::Color32::YELLOW, "⚠ Processing cancelled");

                if ui.button("▶ Process")
                    .on_hover_text("Run BM3D denoising with current parameters")
                    .clicked() {
                    self.start_processing();
                }
            }
            ProcessingState::Error(msg) => {
                ui.colored_label(egui::Color32::RED, format!("✗ Error: {}", msg));

                if ui.button("▶ Retry")
                    .on_hover_text("Retry BM3D processing")
                    .clicked() {
                    self.start_processing();
                }
            }
        }

        ui.add_space(10.0);
        ui.separator();

        // View controls
        ui.heading("View");

        let has_processed = self.processed_volume.is_some();

        // Display mode toggle (Single / Compare)
        ui.horizontal(|ui| {
            ui.label("Display:")
                .on_hover_text("Switch between single image and side-by-side comparison");
            if ui
                .selectable_value(&mut self.display_mode, DisplayMode::Single, "Single")
                .on_hover_text("View one image at a time")
                .clicked()
            {
                self.update_window_level_for_slice();
            }

            ui.add_enabled_ui(has_processed, |ui| {
                if ui
                    .selectable_value(&mut self.display_mode, DisplayMode::Compare, "Compare")
                    .on_hover_text("Show original, processed, and difference side by side")
                    .clicked()
                {
                    self.update_window_level_for_slice();
                }
            });
        });

        // Single view: Original/Processed toggle
        if self.display_mode == DisplayMode::Single {
            ui.horizontal(|ui| {
                ui.label("View:")
                    .on_hover_text("Select which volume to display");
                if ui
                    .selectable_value(&mut self.view_mode, ViewMode::Original, "Original")
                    .on_hover_text("View the original input data")
                    .clicked()
                {
                    self.update_window_level_for_slice();
                }

                ui.add_enabled_ui(has_processed, |ui| {
                    if ui
                        .selectable_value(&mut self.view_mode, ViewMode::Processed, "Processed")
                        .on_hover_text("View the BM3D-processed data")
                        .clicked()
                    {
                        self.update_window_level_for_slice();
                    }
                });
            });
        }

        // Reset view mode if no processed data
        if !has_processed {
            if self.view_mode == ViewMode::Processed {
                self.view_mode = ViewMode::Original;
            }
            if self.display_mode == DisplayMode::Compare {
                self.display_mode = DisplayMode::Single;
            }
        }

        ui.add_space(10.0);
        ui.separator();

        // Save button
        ui.heading("Export");
        if ui.button("💾 Save...")
            .on_hover_text("Save processed data to file")
            .clicked() {
            self.save_dialog.open();
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll processing progress
        self.processing_manager.poll_progress();

        // Check if processing completed and store result
        if matches!(
            self.processing_manager.state(),
            ProcessingState::Completed
        ) {
            if let Some(result) = self.processing_manager.take_processed_result() {
                let processed_vol = Volume3D::new(result);
                if let Some(original) = &self.volume {
                    let mut processed = processed_vol;
                    processed.set_axis_mapping(original.axis_mapping());
                    self.processed_volume = Some(processed);
                } else {
                    self.processed_volume = Some(processed_vol);
                }
                self.view_mode = ViewMode::Processed;
                self.update_window_level_for_slice();
            }
        }

        // Request repaint while processing
        if self.processing_manager.is_processing() {
            ctx.request_repaint();
        }

        // Save dialog
        let has_processed = self.processed_volume.is_some();
        if let Some(request) = self.save_dialog.show(ctx, has_processed) {
            self.handle_save_request(&request);
        }

        // Top panel
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("BM3D Volume Viewer");
                ui.separator();

                if ui.button("📂 Open File")
                    .on_hover_text("Load TIFF stack or HDF5 file")
                    .clicked() {
                    self.open_file_dialog();
                }

                if let Some(path) = &self.file_path {
                    ui.separator();
                    ui.label(format!(
                        "File: {}",
                        path.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("unknown")
                    ));
                }

                if let Some(vol) = self.current_volume() {
                    let shape = vol.original_shape();
                    ui.separator();
                    ui.label(format!("Shape: [{}, {}, {}]", shape[0], shape[1], shape[2]));
                }

                // Display mode indicator
                ui.separator();
                match self.display_mode {
                    DisplayMode::Single => match self.view_mode {
                        ViewMode::Original => ui.label("Viewing: Original"),
                        ViewMode::Processed => {
                            ui.colored_label(egui::Color32::LIGHT_GREEN, "Viewing: Processed")
                        }
                    },
                    DisplayMode::Compare => {
                        ui.colored_label(egui::Color32::LIGHT_BLUE, "Viewing: Compare")
                    }
                };
            });

            if let Some(error) = &self.error_message {
                ui.colored_label(egui::Color32::RED, format!("⚠ {}", error));
            }
        });

        // Left panel for HDF5 tree (if browsing)
        if self.hdf5_tree.is_some() {
            egui::SidePanel::left("hdf5_tree_panel")
                .resizable(true)
                .default_width(300.0)
                .show(ctx, |ui| {
                    ui.heading("Select Dataset");
                    ui.separator();

                    ui.label("Select a 3D dataset to load:");
                    ui.add_space(5.0);

                    let mut dataset_to_load: Option<String> = None;

                    if let Some(tree) = &mut self.hdf5_tree {
                        if let Some(selection) = tree.show(ui) {
                            if selection.shape.len() == 3 {
                                dataset_to_load = Some(selection.dataset_path);
                            } else {
                                self.error_message = Some(format!(
                                    "Dataset must be 3D, got {}D",
                                    selection.shape.len()
                                ));
                            }
                        }
                    }

                    ui.add_space(10.0);
                    if let Some(tree) = &self.hdf5_tree {
                        if let Some(selected) = tree.selected_path() {
                            if ui.button(format!("Load: {}", selected))
                                .on_hover_text("Load the selected dataset for viewing and processing")
                                .clicked() {
                                dataset_to_load = Some(selected.clone());
                            }
                        }
                    }

                    if let Some(path) = dataset_to_load {
                        self.load_hdf5_dataset(&path);
                    }
                });
        }

        // Right panel for processing controls (when volume loaded)
        if self.volume.is_some() {
            egui::SidePanel::right("processing_panel")
                .resizable(true)
                .default_width(250.0)
                .show(ctx, |ui| {
                    self.show_processing_panel(ui);
                });
        }

        // Main central panel
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.current_volume().is_some() {
                // Axis mapping controls
                let mut mapping_changed = false;
                if let Some(widget) = &mut self.axis_mapping_widget {
                    mapping_changed = widget.show(ui);
                }

                // Apply mapping changes to both volumes
                if mapping_changed {
                    if let Some(widget) = &self.axis_mapping_widget {
                        let mapping = widget.mapping();
                        if let Some(vol) = &mut self.volume {
                            vol.set_axis_mapping(mapping);
                        }
                        if let Some(vol) = &mut self.processed_volume {
                            vol.set_axis_mapping(mapping);
                        }
                    }
                    self.compare_view.reset(); // Reset compare view textures
                    self.update_window_level_for_slice();
                }

                ui.separator();

                // Visualization controls row
                ui.horizontal(|ui| {
                    self.colormap_selector.show(ui);
                    ui.separator();
                    self.window_level.show(ui);
                    ui.separator();
                    ui.checkbox(&mut self.keep_aspect_ratio, "1:1 Aspect")
                        .on_hover_text("Maintain equal aspect ratio for image display");
                });

                ui.separator();

                // Display based on mode
                match self.display_mode {
                    DisplayMode::Single => {
                        // Single slice viewer
                        let vol_ref = match self.view_mode {
                            ViewMode::Original => self.volume.as_ref(),
                            ViewMode::Processed => {
                                self.processed_volume.as_ref().or(self.volume.as_ref())
                            }
                        };

                        if let Some(vol) = vol_ref {
                            let colormap = self.colormap_selector.current();
                            let slice_index = self.slice_viewer.current_slice();

                            // Get slice data for histogram and cursor tracking
                            let slice_data = vol.get_slice(slice_index);

                            // Show histogram panel
                            self.single_histogram.show(ui, slice_data.as_ref(), slice_index);

                            // Show slice viewer with cursor tracking
                            let slice_changed =
                                self.slice_viewer.show_with_slice_data(
                                    ui, vol, slice_data.as_ref(), colormap, &self.window_level, self.keep_aspect_ratio
                                );

                            if slice_changed {
                                self.update_window_level_for_slice();
                            }
                        }
                    }
                    DisplayMode::Compare => {
                        // Three-panel comparison view
                        if let (Some(orig), Some(proc)) = (&self.volume, &self.processed_volume) {
                            let colormap = self.colormap_selector.current();
                            let slice_index = self.compare_view.current_slice();

                            // Get slice data for histograms and cursor tracking
                            let orig_slice = orig.get_slice(slice_index);
                            let proc_slice = proc.get_slice(slice_index);

                            // Show histogram panel
                            self.compare_histogram.show(
                                ui,
                                orig_slice.as_ref(),
                                proc_slice.as_ref(),
                                slice_index,
                            );

                            // Show compare view with cursor tracking
                            let slice_changed =
                                self.compare_view.show_with_slice_data(
                                    ui, orig, proc,
                                    orig_slice.as_ref(), proc_slice.as_ref(),
                                    colormap, &self.window_level, self.keep_aspect_ratio
                                );

                            if slice_changed {
                                self.update_window_level_for_slice();
                            }
                        }
                    }
                }
            } else if self.hdf5_tree.is_none() {
                ui.vertical_centered(|ui| {
                    ui.add_space(100.0);
                    ui.heading("No volume loaded");
                    ui.add_space(20.0);
                    ui.label("Click 'Open File' to load a TIFF stack or HDF5 file");
                    ui.add_space(10.0);
                    ui.label("Supported formats:");
                    ui.label("• Multi-page TIFF (.tif, .tiff)");
                    ui.label("• HDF5 (.h5, .hdf5, .nxs)");
                    ui.add_space(20.0);
                    ui.label("Controls:");
                    ui.label("• Mouse wheel: Zoom in/out");
                    ui.label("• Click + drag: Pan image");
                    ui.label("• Slider: Navigate slices");
                });
            }
        });
    }
}
