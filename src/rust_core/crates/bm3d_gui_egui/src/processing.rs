use bm3d_core::{
    bm3d_ring_artifact_removal, fourier_svd::fourier_svd_removal, multiscale_bm3d_streak_removal,
    Bm3dConfig, MultiscaleConfig, RingRemovalMode,
};
use ndarray::{Array2, Array3, Axis};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Progress update from processing thread.
#[derive(Debug, Clone)]
pub enum ProcessingProgress {
    /// Processing started
    Started { total_slices: usize },
    /// Slice completed
    SliceComplete {
        slice_index: usize,
        total_slices: usize,
    },
    /// Processing finished successfully
    Finished { result: Array3<f32> },
    /// Processing was cancelled
    Cancelled,
    /// Error occurred
    Error(String),
}

/// Processing state for UI.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ProcessingState {
    #[default]
    Idle,
    Processing {
        current_slice: usize,
        total_slices: usize,
    },
    Completed,
    Cancelled,
    Error(String),
}

/// Manager for background BM3D processing.
pub struct ProcessingManager {
    /// Current processing state
    state: ProcessingState,
    /// Channel to receive progress updates
    progress_rx: Option<Receiver<ProcessingProgress>>,
    /// Cancel flag shared with worker thread
    cancel_flag: Arc<AtomicBool>,
    /// Handle to worker thread
    worker_handle: Option<JoinHandle<()>>,
    /// Processed result
    processed_result: Option<Array3<f32>>,
}

impl Default for ProcessingManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessingManager {
    pub fn new() -> Self {
        Self {
            state: ProcessingState::Idle,
            progress_rx: None,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            worker_handle: None,
            processed_result: None,
        }
    }

    pub fn state(&self) -> &ProcessingState {
        &self.state
    }

    pub fn is_processing(&self) -> bool {
        matches!(self.state, ProcessingState::Processing { .. })
    }

    pub fn processed_result(&self) -> Option<&Array3<f32>> {
        self.processed_result.as_ref()
    }

    pub fn take_processed_result(&mut self) -> Option<Array3<f32>> {
        self.processed_result.take()
    }

    /// Start processing the input volume.
    ///
    /// `slice_axis` specifies which axis to iterate over for processing.
    /// Each 2D slice perpendicular to this axis will be processed as a sinogram.
    /// For tomography data [angles, Y, X], use slice_axis=1 (Y) to process
    /// each [angles, X] sinogram.
    pub fn start_processing(
        &mut self,
        input: Array3<f32>,
        mode: RingRemovalMode,
        config: MultiscaleConfig<f32>,
        use_multiscale: bool,
        slice_axis: usize,
    ) {
        // Cancel any existing processing
        self.cancel();

        // Reset state
        self.cancel_flag = Arc::new(AtomicBool::new(false));
        self.processed_result = None;

        // Create channel for progress updates
        let (tx, rx) = channel();
        self.progress_rx = Some(rx);

        // Clone cancel flag for worker
        let cancel_flag = self.cancel_flag.clone();

        // Spawn worker thread
        let handle = thread::spawn(move || {
            process_volume_worker(
                input,
                mode,
                config,
                use_multiscale,
                slice_axis,
                tx,
                cancel_flag,
            );
        });

        self.worker_handle = Some(handle);
        self.state = ProcessingState::Processing {
            current_slice: 0,
            total_slices: 0,
        };
    }

    /// Request cancellation of current processing.
    pub fn cancel(&mut self) {
        self.cancel_flag.store(true, Ordering::SeqCst);

        // Wait for worker to finish
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }

        self.progress_rx = None;
    }

    /// Poll for progress updates. Call this each frame.
    pub fn poll_progress(&mut self) {
        // Collect all pending messages first
        let mut messages = Vec::new();
        if let Some(rx) = &self.progress_rx {
            while let Ok(progress) = rx.try_recv() {
                messages.push(progress);
            }
        }

        // Track if we need to clean up
        let mut should_cleanup = false;

        // Process collected messages
        for progress in messages {
            match progress {
                ProcessingProgress::Started { total_slices } => {
                    self.state = ProcessingState::Processing {
                        current_slice: 0,
                        total_slices,
                    };
                }
                ProcessingProgress::SliceComplete {
                    slice_index,
                    total_slices,
                } => {
                    self.state = ProcessingState::Processing {
                        current_slice: slice_index + 1,
                        total_slices,
                    };
                }
                ProcessingProgress::Finished { result } => {
                    self.processed_result = Some(result);
                    self.state = ProcessingState::Completed;
                    should_cleanup = true;
                }
                ProcessingProgress::Cancelled => {
                    self.state = ProcessingState::Cancelled;
                    should_cleanup = true;
                }
                ProcessingProgress::Error(msg) => {
                    self.state = ProcessingState::Error(msg);
                    should_cleanup = true;
                }
            }
        }

        // Clean up after processing all messages
        if should_cleanup {
            self.progress_rx = None;
            self.worker_handle = None;
        }
    }

    /// Reset state to idle.
    pub fn reset(&mut self) {
        self.cancel();
        self.state = ProcessingState::Idle;
        self.processed_result = None;
    }
}

/// Worker function that runs in background thread.
/// Processes the volume by iterating over `slice_axis` and applying BM3D to each 2D slice.
fn process_volume_worker(
    input: Array3<f32>,
    mode: RingRemovalMode,
    config: MultiscaleConfig<f32>,
    use_multiscale: bool,
    slice_axis: usize,
    tx: Sender<ProcessingProgress>,
    cancel_flag: Arc<AtomicBool>,
) {
    let shape = input.shape();
    let num_slices = shape[slice_axis];

    // Send start message
    if tx
        .send(ProcessingProgress::Started {
            total_slices: num_slices,
        })
        .is_err()
    {
        return;
    }

    // Allocate output with same shape as input
    let mut output = Array3::<f32>::zeros(input.raw_dim());

    // Process slice by slice along the specified axis
    for slice_idx in 0..num_slices {
        // Check for cancellation
        if cancel_flag.load(Ordering::SeqCst) {
            let _ = tx.send(ProcessingProgress::Cancelled);
            return;
        }

        // Extract 2D slice along the slice_axis
        let slice_2d = input.index_axis(Axis(slice_axis), slice_idx);

        // Convert to owned Array2 for processing
        let slice_owned: Array2<f32> = slice_2d.to_owned();

        // Process slice
        let result = match mode {
            // Generic mode always uses standard BM3D
            RingRemovalMode::Generic => {
                bm3d_ring_artifact_removal(slice_owned.view(), mode, &config.bm3d_config)
            }
            // Streak mode can use Multiscale if enabled via use_multiscale flag
            RingRemovalMode::Streak => {
                if use_multiscale {
                    multiscale_bm3d_streak_removal(slice_owned.view(), &config)
                } else {
                    bm3d_ring_artifact_removal(slice_owned.view(), mode, &config.bm3d_config)
                }
            }
            // MultiscaleStreak mode always uses multiscale processing
            RingRemovalMode::MultiscaleStreak => {
                multiscale_bm3d_streak_removal(slice_owned.view(), &config)
            }
            RingRemovalMode::FourierSvd => {
                // Fourier-SVD requires f64 for precision
                let slice_f64 = slice_owned.mapv(|x| x as f64);
                let fft_alpha = config.bm3d_config.fft_alpha as f64;
                let notch_width = config.bm3d_config.notch_width as f64;
                let result_f64 = fourier_svd_removal(slice_f64.view(), fft_alpha, notch_width);
                // Convert back to f32
                Ok(result_f64.mapv(|x| x as f32))
            }
        };

        match result {
            Ok(processed) => {
                // Copy result back to output at the correct position
                let mut out_slice = output.index_axis_mut(Axis(slice_axis), slice_idx);
                out_slice.assign(&processed);
            }
            Err(e) => {
                let _ = tx.send(ProcessingProgress::Error(format!(
                    "Failed to process slice {}: {}",
                    slice_idx, e
                )));
                return;
            }
        }

        // Send progress update
        if tx
            .send(ProcessingProgress::SliceComplete {
                slice_index: slice_idx,
                total_slices: num_slices,
            })
            .is_err()
        {
            return;
        }
    }

    // Send completion
    let _ = tx.send(ProcessingProgress::Finished { result: output });
}
