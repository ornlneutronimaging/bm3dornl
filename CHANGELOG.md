# Changelog

## Unreleased

### Fixed

- GUI: loading HDF5 volumes is back at bulk-read speed. The load progress bar introduced in 0.10.0's follow-up work read the dataset slice by slice, and on gzip-chunked datasets every chunk spans several slices, so each chunk was decompressed once per slice it covers — measured 16x slower than a bulk read on the chunked test volume (~62 s vs ~4 s for 591 slices). The loader now reads chunk-aligned blocks of slices, so each chunk decompresses exactly once and the progress bar is kept. Loaded volumes are identical to the previous reader's, pinned by a test on a chunked file whose slice count is not a multiple of the block.
- Fourier-SVD: `fourier_svd_removal` (and the `FourierSvd` ring-removal mode that calls it) no longer returns flat-background sinograms unchanged. Its magnitude gate removes detail below a threshold and protects detail above it, and the threshold comes from a MAD-based scale of the rank-1 column profile. MAD is the median of the absolute deviations, so when most columns see numerically flat air — the normal case for simulated phantoms — it collapses to machine residue (about 1e-11 on the shipped test sinogram), the gate removes nothing, and the call returns its input with no influence from `fft_alpha` or `notch_width` (#133). That specific degeneracy is now detected (threshold below 1e-6 of the largest deviation, a regime the gate cannot act in), and the scale is re-estimated over the entries above the numerical floor. The rescue is guarded: it declines and keeps the no-op unless at least a quarter of the columns carry information, and unless the resulting correction stays under 10% of the input's energy — a correction larger than that would be removing the sample, not streaks.
- Every input outside that degeneracy is byte-identical to the previous release, verified output-for-output against the previous build across the full measured CG-1D volume and the noise-bearing synthetic cases. 533 of the volume's 540 sinograms take the original path unchanged; the remaining 7 have a column profile with exactly zero median deviation (the median filter returns an element of its window, so on a smooth profile more than half the detail entries can be exactly zero), where the previous code substituted a standard deviation and measurably degraded the output against the volume's reference reconstruction. No scale is estimable from zero dispersion, so those are now returned unchanged — closer to the reference than either previous behaviour.
- GUI: the Multiscale Streak panel gains an "Input is already log-transformed (attenuation)" toggle, wired to the multiscale pipeline's `log_domain_input`. Without it the GUI could only run the linear-transmission path, which applies a second logarithm to attenuation sinograms and produces unusable output on the usual normalized CG-1D data.
- GUI: cancelling a processing run no longer strands the app. The cancel path joined the worker thread and then dropped the progress channel unread, so the worker's final "cancelled" message never reached the state machine — the progress bar froze mid-run, the Process button never returned, and only loading a new file recovered (#136). The cancel path now drains the channel after the worker exits (so a run that finished just before the request still delivers its result), reports a worker that died without a result as an error, and otherwise lands in the cancelled state, whose existing UI offers Process again as intended.
- GUI: patch size is now a slider covering 4-16 instead of a menu offering only 4, 8, or 16. The core accepts any patch size (only the optional 8x8 Hadamard fast mode is size-specific, and it simply stays off for other sizes), and the documentation recommends 7 or 8 — a value the menu made unreachable (#135).
- Multiscale: the multiscale streak path was reworked to follow Mäkinen et al. (2021) more closely. It now denoises in the log domain the model is defined on, on a vertically binned sinogram, as overlapping segments with a locally estimated sigma, with each pyramid level given the noise spectrum its own residual actually has; the automatic noise estimate measures the streak amplitude instead of the air-region floor, which previously collapsed to quantization residue on flat-background inputs and silently turned the whole correction into a no-op. The input is expected to be linear transmission data (the pipeline applies and undoes the logarithm itself); data that is already log-transformed — attenuation sinograms, the normal CG-1D product — must be declared with `log_domain_input=True` (Python) or the matching GUI toggle, otherwise it is logged a second time and the result is unusable. Input with no positive values is rejected with a message naming that flag (a constant blank slice passes through unchanged). The rework touches only the multiscale path: single-scale streak, generic, and Fourier-SVD behave exactly as they did before it (Fourier-SVD's own change is the separate entry above).

### Changed

- The `bm3dornl.plot` module was removed and its single function, `compute_cdf`, relocated to `bm3dornl.utils` (#138). The module never contained plotting code — `compute_cdf` computes an image's cumulative distribution function, a diagnostic that fits the utils module's purpose. Anyone importing `from bm3dornl.plot import compute_cdf` must switch to `from bm3dornl.utils import compute_cdf`; the function itself is unchanged.
- Fourier-SVD output changes on flat-background inputs as a consequence of the fix above. Any previously recorded Fourier-SVD quality figures for such inputs were measured on an essentially unmodified array and should be regenerated. On the benchmark phantom, SSIM against ground truth rises from 0.9510 — the unprocessed input's own score — to 0.9737.

### Documentation

- `estimate_noise_sigma`: the docstring claimed the function estimates the image's noise standard deviation and demonstrated it on i.i.d. Gaussian noise, promising a result "close to 0.1" that the function does not produce (it returns ~0.013 there). The estimator measures the amplitude of vertical streaks — its vertical Gaussian pre-filter deliberately suppresses pixel-level i.i.d. noise — and is the same estimator the pipeline uses to fill in `sigma_random` when it is set to 0.0. The Python docstring, the Rust doc comments, and the crate README now state that contract, the example constructs actual streak noise, and new tests pin both behaviours (#134). No behaviour changed.

### Known limitation

- When a sinogram's background is both the majority of columns and carries noise, the MAD scale reports that noise floor, and streaks larger than the background noise are still under-corrected. This regime is unchanged by the fix (outputs remain byte-identical to the previous release there); correcting it needs a reliable separation of sample columns from background columns, which is follow-up work.

## 0.10.0 - 2026-06-17

### Changed

- Python: the default `mode` for `bm3d_ring_artifact_removal` changed from `"generic"` to `"streak"`, so the API default matches the package's ring-artifact-removal focus and the GUI/documentation defaults (#115).
- Python: the `threshold` parameter default changed from `2.7` to `None`. When `None`, the backend default is applied automatically: `2.7` for single-scale BM3D and `3.5` for multi-scale BM3D. Pass an explicit value to override (#115).
- Rust core: the default `RingRemovalMode` changed from `MultiscaleStreak` to `Streak`, and the default `sigma_random` changed from `0.0` (auto-estimate) to `0.1`; the multi-scale default debinning iterations increased from 10 to 30 (#115).
- GUI: BM3D parameter defaults are now derived from the Rust `Bm3dConfig`/`MultiscaleConfig` defaults instead of being hard-coded, so the GUI, Python API, and Rust core stay in sync (mode now defaults to Streak; `sigma_random` to 0.1; `max_matches` to 16) (#115).
- Docs/README: corrected the parameter reference table to match the actual backend defaults (`step_size` 3→4, `search_window` 39→24, `streak_sigma_smooth` 1.0→3.0) and documented the new `threshold=None` / dual-default behavior (#115).

### Fixed

- GUI: large TIFF stack exports that exceed the classic TIFF ~4 GiB offset limit are now written as BigTIFF automatically (with a 64 MiB safety margin), fixing failures when saving large volumes; payload size is computed with checked arithmetic and slices are written via borrowed memory where possible (#115, closes #113).

### Security

- Bumped pyo3 and the Rust `numpy` crate from 0.25 to 0.29 to clear two Dependabot advisories: a HIGH-severity out-of-bounds read in `nth`/`nth_back` for `PyList`/`PyTuple`, and a MEDIUM-severity missing `Sync` bound on `PyCFunction::new_closure` closures (#112).
- Bumped the transitive `rand` crate (0.8.5→0.8.6 and 0.9.2→0.9.3) to resolve a soundness advisory, GHSA-cq8v-f236-94qc (#111).

### Internal

- Rust: upgraded the workspace from edition 2021 to edition 2024 and applied the resulting `cargo fix`/clippy migrations (#95, #107).
- Rust: finished the edition-2024 `unsafe_op_in_unsafe_fn` migration in `bm3d_core` by wrapping the SSE2/AVX2 SIMD bodies in explicit `unsafe {}` blocks with SAFETY notes and removing the `#[allow(...)]` shims (#110).
- GUI: re-enabled tests for the `bm3dornl-gui` binary (removed `test = false`) and added unit/round-trip tests for the TIFF/BigTIFF export path (#115).
- Tests: added `tests/test_defaults.py` covering the new Python default `mode`/`threshold` behavior (#115).
- CI/docs: bumped `prefix-dev/setup-pixi` (0.9.4→0.9.5→0.9.6) and `softprops/action-gh-release` (v2→v3), and upgraded the Read the Docs build image to ubuntu-24.04 (#104, #105, #106, #108).
