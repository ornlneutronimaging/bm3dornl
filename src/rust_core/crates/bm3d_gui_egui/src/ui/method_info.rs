//! "About this method" panel: a short explanation of each ring-removal
//! algorithm, with links to the implementing source file and to the
//! literature it is based on.

use bm3d_core::RingRemovalMode;
use eframe::egui;

/// Base URL for source links (pinned to the `main` branch so links stay valid).
const SOURCE_BASE: &str = "https://github.com/ornlneutronimaging/bm3dornl/blob/main/";

/// A literature reference shown as a clickable DOI link.
struct Reference {
    /// Short citation, e.g. "Dabov et al. 2007, IEEE TIP 16(8)"
    citation: &'static str,
    /// Title of the paper
    title: &'static str,
    /// Resolvable URL (DOI preferred)
    url: &'static str,
}

/// Static description of one processing method.
struct MethodInfo {
    /// One-line summary shown at the top of the panel
    summary: &'static str,
    /// Bullet points describing how the algorithm works
    steps: &'static [&'static str],
    /// When to prefer this method
    use_when: &'static str,
    /// Source files implementing the method, relative to the repo root
    sources: &'static [&'static str],
    /// Literature the implementation is based on or related to
    references: &'static [Reference],
}

const BM3D_2007: Reference = Reference {
    citation: "Dabov, Foi, Katkovnik & Egiazarian (2007), IEEE Trans. Image Process. 16(8)",
    title: "Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering",
    url: "https://doi.org/10.1109/TIP.2007.901238",
};

const CORRELATED_NOISE_2020: Reference = Reference {
    citation: "Mäkinen, Azzari & Foi (2020), IEEE Trans. Image Process. 29",
    title: "Collaborative Filtering of Correlated Noise: Exact Transform-Domain Variance \
            for Improved Shrinkage and Patch Matching",
    url: "https://doi.org/10.1109/TIP.2020.3014721",
};

const MULTISCALE_2021: Reference = Reference {
    citation: "Mäkinen, Marchesini & Foi (2021), J. Synchrotron Rad. 28(3)",
    title: "Ring artifact reduction via multiscale nonlocal collaborative filtering \
            of spatially correlated noise",
    url: "https://doi.org/10.1107/S1600577521001910",
};

const WAVELET_FOURIER_2009: Reference = Reference {
    citation: "Münch, Trtik, Marone & Stampanoni (2009), Opt. Express 17(10)",
    title: "Stripe and ring artifact removal with combined wavelet-Fourier filtering",
    url: "https://doi.org/10.1364/OE.17.008567",
};

const RING_SVD_2018: Reference = Reference {
    citation: "Vo, Atwood & Drakopoulos (2018), Opt. Express 26(22)",
    title: "Superior techniques for eliminating ring artifacts in X-ray micro-tomography",
    url: "https://doi.org/10.1364/OE.26.028396",
};

const GENERIC_INFO: MethodInfo = MethodInfo {
    summary: "Standard BM3D denoising, assuming white (uncorrelated) Gaussian noise.",
    steps: &[
        "Block matching: for every reference patch, find similar patches in a search window \
         and stack them into a 3D group.",
        "Hard-thresholding pass: transform each group (2D DCT/Hadamard + 1D Haar), zero the \
         small coefficients, invert, and aggregate with adaptive weights.",
        "Wiener pass: repeat the grouping on the first estimate and apply empirical Wiener \
         shrinkage using the noise power spectrum (scalar sigma).",
    ],
    use_when: "Random pixel noise without a dominant streak/ring structure. Not designed for \
               ring artifacts: use Streak or Multiscale Streak for sinograms.",
    sources: &[
        "src/rust_core/crates/bm3d_core/src/orchestration.rs",
        "src/rust_core/crates/bm3d_core/src/pipeline.rs",
        "src/rust_core/crates/bm3d_core/src/block_matching.rs",
    ],
    references: &[BM3D_2007],
};

const STREAK_INFO: MethodInfo = MethodInfo {
    summary: "Single-scale BM3D tuned for vertical streaks in sinograms (rings after \
              reconstruction).",
    steps: &[
        "Streak pre-subtraction: iteratively estimate the static per-column streak profile \
         (Gaussian smoothing + median filtering) and remove it before denoising.",
        "Anisotropic PSD: model the remaining streak noise as correlated along the vertical \
         axis, so the transform-domain shrinkage targets streak energy rather than image \
         structure.",
        "Run the two BM3D passes (hard threshold + Wiener) with this PSD, then denormalize.",
    ],
    use_when: "The default choice for narrow to medium ring artifacts in sinograms.",
    sources: &[
        "src/rust_core/crates/bm3d_core/src/orchestration.rs",
        "src/rust_core/crates/bm3d_core/src/streak.rs",
        "src/rust_core/crates/bm3d_core/src/pipeline.rs",
    ],
    references: &[BM3D_2007, CORRELATED_NOISE_2020, MULTISCALE_2021],
};

const MULTISCALE_STREAK_INFO: MethodInfo = MethodInfo {
    summary: "Multi-scale BM3D streak removal following Mäkinen, Marchesini & Foi (2021).",
    steps: &[
        "Build a horizontal pyramid of the sinogram by sum-convolution binning along the \
         detector axis.",
        "Process coarse-to-fine: run Streak-mode BM3D at each scale and propagate the \
         denoised residual to the next finer level.",
        "Debin with cubic spline interpolation so that wide streaks captured at coarse \
         scales are removed at full resolution.",
    ],
    use_when: "Wide ring artifacts that single-scale BM3D cannot capture within one patch. \
               Slower than Streak; increase Scales for wider streaks.",
    sources: &[
        "src/rust_core/crates/bm3d_core/src/multiscale.rs",
        "src/rust_core/crates/bm3d_core/src/orchestration.rs",
    ],
    references: &[MULTISCALE_2021, CORRELATED_NOISE_2020, BM3D_2007],
};

const FOURIER_SVD_INFO: MethodInfo = MethodInfo {
    summary: "Fast two-stage destriping: FFT-guided energy detection followed by a rank-1 \
              SVD streak model with magnitude gating (about 2.6x faster than BM3D).",
    steps: &[
        "Stage 1 (FFT): isolate near-vertical frequencies with a Gaussian notch filter, \
         compute a per-column streak energy profile, and use it to modulate the removal \
         threshold (FFT Alpha, Notch Width).",
        "Stage 2 (SVD): extract the first principal component by power iteration, median \
         filter it to separate baseline from streak detail, and soft-gate its magnitude.",
        "Reconstruct the streak as a rank-1 outer product and subtract it from the input.",
    ],
    use_when: "Subtle or low-contrast streaks, or when speed matters. Preserves structure well \
               at high SNR. This method is original to bm3dornl; the references below cover \
               the Fourier-notch and SVD ideas it builds on.",
    sources: &[
        "src/rust_core/crates/bm3d_core/src/fourier_svd.rs",
        "src/bm3dornl/fourier_svd.py",
    ],
    references: &[WAVELET_FOURIER_2009, RING_SVD_2018],
};

fn info_for(mode: RingRemovalMode) -> &'static MethodInfo {
    match mode {
        RingRemovalMode::Generic => &GENERIC_INFO,
        RingRemovalMode::Streak => &STREAK_INFO,
        RingRemovalMode::MultiscaleStreak => &MULTISCALE_STREAK_INFO,
        RingRemovalMode::FourierSvd => &FOURIER_SVD_INFO,
    }
}

/// Show a collapsible "About this method" section for the given mode.
pub fn show_method_info(ui: &mut egui::Ui, mode: RingRemovalMode) {
    let info = info_for(mode);

    egui::CollapsingHeader::new("About this method")
        .id_salt("method_info")
        .default_open(false)
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 4.0;

            ui.label(egui::RichText::new(info.summary).strong());

            ui.add_space(2.0);
            ui.label("How it works:");
            for step in info.steps {
                ui.horizontal_wrapped(|ui| {
                    ui.label("•");
                    ui.label(*step);
                });
            }

            ui.add_space(2.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Use when:").italics());
                ui.label(info.use_when);
            });

            ui.add_space(4.0);
            ui.label("Source:");
            for path in info.sources {
                let file_name = path.rsplit('/').next().unwrap_or(path);
                ui.horizontal_wrapped(|ui| {
                    ui.label("•");
                    ui.hyperlink_to(file_name, format!("{SOURCE_BASE}{path}"))
                        .on_hover_text(*path);
                });
            }

            ui.add_space(4.0);
            ui.label("Literature:");
            for reference in info.references {
                ui.horizontal_wrapped(|ui| {
                    ui.label("•");
                    ui.hyperlink_to(reference.citation, reference.url)
                        .on_hover_text(format!("{}\n{}", reference.title, reference.url));
                });
            }
        });
}
