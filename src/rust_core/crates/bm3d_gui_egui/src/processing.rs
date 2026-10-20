use bm3d_core::orchestration::bm3d_ring_artifact_removal_with_plans;
use bm3d_core::{
    Bm3dConfig, Bm3dPlans, MultiscaleConfig, RingRemovalMode, bm3d_ring_artifact_removal,
    estimate_noise_sigma, fourier_svd::fourier_svd_removal, multiscale_bm3d_streak_removal,
    multiscale_bm3d_streak_removal_with_plans,
};
use ndarray::{Array2, Array3, Axis};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::{self, JoinHandle};

const REUSE_VOLUME_SIGMA_ENV: &str = "BM3D_REUSE_VOLUME_SIGMA";

fn resolve_reuse_volume_sigma() -> bool {
    std::env::var(REUSE_VOLUME_SIGMA_ENV)
        .ok()
        .map(|value| {
            let v = value.trim();
            v == "1"
                || v.eq_ignore_ascii_case("true")
                || v.eq_ignore_ascii_case("yes")
                || v.eq_ignore_ascii_case("on")
        })
        .unwrap_or(false)
}

#[derive(Debug, Clone)]
pub enum ProcessingProgress {
    Started {
        total_slices: usize,
    },
    SliceComplete {
        slice_index: usize,
        total_slices: usize,
    },
    Finished {
        result: Array3<f32>,
    },
    Cancelled,
    Error(String),
}

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

pub struct ProcessingManager {
    state: ProcessingState,
    progress_rx: Option<Receiver<ProcessingProgress>>,
    cancel_flag: Arc<AtomicBool>,
    worker_handle: Option<JoinHandle<()>>,
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

    pub fn start_processing(
        &mut self,
        input: Array3<f32>,
        mode: RingRemovalMode,
        config: MultiscaleConfig<f32>,
        use_multiscale: bool,
        slice_axis: usize,
    ) {
        self.cancel();

        self.cancel_flag = Arc::new(AtomicBool::new(false));
        self.processed_result = None;

        let (tx, rx) = channel();
        self.progress_rx = Some(rx);

        let cancel_flag = self.cancel_flag.clone();

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

    pub fn cancel(&mut self) {
        self.cancel_flag.store(true, Ordering::SeqCst);

        let join_result = self.worker_handle.take().map(|handle| handle.join());

        if let Some(rx) = self.progress_rx.take() {
            while let Ok(progress) = rx.try_recv() {
                self.apply_progress(progress);
            }
        }

        if join_result.is_some() && matches!(self.state, ProcessingState::Processing { .. }) {
            self.state = match join_result {
                Some(Err(_)) => ProcessingState::Error(
                    "processing thread panicked before reporting a result".to_string(),
                ),
                _ => ProcessingState::Cancelled,
            };
        }
    }

    /// Returns true when the message is terminal (worker finished, cancelled, or errored).
    fn apply_progress(&mut self, progress: ProcessingProgress) -> bool {
        match progress {
            ProcessingProgress::Started { total_slices } => {
                self.state = ProcessingState::Processing {
                    current_slice: 0,
                    total_slices,
                };
                false
            }
            ProcessingProgress::SliceComplete {
                slice_index,
                total_slices,
            } => {
                self.state = ProcessingState::Processing {
                    current_slice: slice_index + 1,
                    total_slices,
                };
                false
            }
            ProcessingProgress::Finished { result } => {
                self.processed_result = Some(result);
                self.state = ProcessingState::Completed;
                true
            }
            ProcessingProgress::Cancelled => {
                self.state = ProcessingState::Cancelled;
                true
            }
            ProcessingProgress::Error(msg) => {
                self.state = ProcessingState::Error(msg);
                true
            }
        }
    }

    /// Call this each frame.
    pub fn poll_progress(&mut self) {
        let mut messages = Vec::new();
        if let Some(rx) = &self.progress_rx {
            while let Ok(progress) = rx.try_recv() {
                messages.push(progress);
            }
        }

        let mut should_cleanup = false;

        for progress in messages {
            should_cleanup |= self.apply_progress(progress);
        }

        if should_cleanup {
            self.progress_rx = None;
            self.worker_handle = None;
        }
    }

    pub fn reset(&mut self) {
        self.cancel();
        self.state = ProcessingState::Idle;
        self.processed_result = None;
    }
}

fn process_volume_worker(
    input: Array3<f32>,
    mode: RingRemovalMode,
    config: MultiscaleConfig<f32>,
    use_multiscale: bool,
    slice_axis: usize,
    tx: Sender<ProcessingProgress>,
    cancel_flag: Arc<AtomicBool>,
) {
    let mut config = config;
    let shape = input.shape();
    let num_slices = shape[slice_axis];
    let shared_plans = if mode != RingRemovalMode::FourierSvd {
        Some(Bm3dPlans::new(
            config.bm3d_config.patch_size,
            config.bm3d_config.max_matches,
        ))
    } else {
        None
    };
    if mode != RingRemovalMode::FourierSvd
        && config.bm3d_config.sigma_random <= 1e-6
        && num_slices > 0
        && resolve_reuse_volume_sigma()
    {
        let first_slice = input.index_axis(Axis(slice_axis), 0);
        let estimated = estimate_noise_sigma(first_slice).max(1e-3);
        config.bm3d_config.sigma_random = estimated;
    }

    if tx
        .send(ProcessingProgress::Started {
            total_slices: num_slices,
        })
        .is_err()
    {
        return;
    }

    let mut output = Array3::<f32>::zeros(input.raw_dim());
    let mut slice_buffer = if num_slices > 0 {
        let first_slice = input.index_axis(Axis(slice_axis), 0);
        Array2::<f32>::zeros(first_slice.dim())
    } else {
        Array2::<f32>::zeros((0, 0))
    };

    for slice_idx in 0..num_slices {
        if cancel_flag.load(Ordering::SeqCst) {
            let _ = tx.send(ProcessingProgress::Cancelled);
            return;
        }

        let slice_2d = input.index_axis(Axis(slice_axis), slice_idx);

        slice_buffer.assign(&slice_2d);
        let slice_view = slice_buffer.view();

        let result = match mode {
            RingRemovalMode::Generic => {
                if let Some(plans) = shared_plans.as_ref() {
                    bm3d_ring_artifact_removal_with_plans(
                        slice_view,
                        mode,
                        &config.bm3d_config,
                        plans,
                    )
                } else {
                    bm3d_ring_artifact_removal(slice_view, mode, &config.bm3d_config)
                }
            }
            RingRemovalMode::Streak => {
                if use_multiscale {
                    if let Some(plans) = shared_plans.as_ref() {
                        multiscale_bm3d_streak_removal_with_plans(slice_view, &config, plans)
                    } else {
                        multiscale_bm3d_streak_removal(slice_view, &config)
                    }
                } else if let Some(plans) = shared_plans.as_ref() {
                    bm3d_ring_artifact_removal_with_plans(
                        slice_view,
                        mode,
                        &config.bm3d_config,
                        plans,
                    )
                } else {
                    bm3d_ring_artifact_removal(slice_view, mode, &config.bm3d_config)
                }
            }
            RingRemovalMode::MultiscaleStreak => {
                if let Some(plans) = shared_plans.as_ref() {
                    multiscale_bm3d_streak_removal_with_plans(slice_view, &config, plans)
                } else {
                    multiscale_bm3d_streak_removal(slice_view, &config)
                }
            }
            RingRemovalMode::FourierSvd => {
                let slice_f64 = slice_view.mapv(|x| x as f64);
                let fft_alpha = config.bm3d_config.fft_alpha as f64;
                let notch_width = config.bm3d_config.notch_width as f64;
                let result_f64 = fourier_svd_removal(slice_f64.view(), fft_alpha, notch_width);
                Ok(result_f64.mapv(|x| x as f32))
            }
        };

        match result {
            Ok(processed) => {
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

    let _ = tx.send(ProcessingProgress::Finished { result: output });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> MultiscaleConfig<f32> {
        let mut config = MultiscaleConfig::<f32>::default();
        // Explicit sigma: keep the worker off the auto-estimation path.
        config.bm3d_config.sigma_random = 0.1;
        config
    }

    #[test]
    fn cancel_leaves_an_actionable_state() {
        let volume = Array3::<f32>::from_elem((64, 64, 65), 0.5);
        let mut manager = ProcessingManager::new();
        manager.start_processing(volume, RingRemovalMode::Streak, test_config(), false, 0);
        manager.cancel();
        assert!(
            !manager.is_processing(),
            "cancel left the manager stuck in Processing: {:?}",
            manager.state()
        );
        assert!(matches!(
            manager.state(),
            ProcessingState::Cancelled | ProcessingState::Completed
        ));
    }

    #[test]
    fn cancel_after_completion_keeps_the_result() {
        let volume = Array3::<f32>::from_elem((2, 32, 33), 0.5);
        let mut manager = ProcessingManager::new();
        manager.start_processing(volume, RingRemovalMode::Streak, test_config(), false, 0);
        // Wait for the worker to finish before cancelling,
        // so the drain (not the cancel flag) decides the outcome.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while manager
            .worker_handle
            .as_ref()
            .is_some_and(|handle| !handle.is_finished())
            && std::time::Instant::now() < deadline
        {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        manager.cancel();
        assert!(matches!(manager.state(), ProcessingState::Completed));
        assert!(manager.processed_result().is_some());
    }

    #[test]
    fn poll_reaches_completed_and_delivers_result() {
        let volume = Array3::<f32>::from_elem((2, 32, 33), 0.5);
        let mut manager = ProcessingManager::new();
        manager.start_processing(volume, RingRemovalMode::Streak, test_config(), false, 0);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while manager.is_processing() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
            manager.poll_progress();
        }
        assert!(matches!(manager.state(), ProcessingState::Completed));
        assert!(manager.take_processed_result().is_some());
    }
}
