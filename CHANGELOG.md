# Changelog

## Unreleased

### Fixed

- GUI HDF5 volumes load at bulk-read speed again. The loader now reads gzip
  data in chunk-aligned slice blocks. Each chunk decompresses once while load
  progress remains visible. A 591-slice test volume loads in about 4 seconds.
  Slice-by-slice loading took about 62 seconds. This is about 16 times faster.
  Loaded values remain identical to the previous reader. A test covers a final
  block with fewer slices.
- Fourier-SVD now removes streaks from flat-background sinograms (#133). The
  fix covers `fourier_svd_removal` and the `FourierSvd` mode. Previously, the
  threshold used the median absolute deviation (MAD). MAD measures the median
  distance from a set's median. Flat air columns reduced MAD to floating-point
  rounding values. The shipped test sinogram produced about 1e-11. The gate
  then protected all details. Neither `fft_alpha` nor `notch_width` affected
  the output. The method now detects thresholds below 1e-6 of the largest
  deviation. It then estimates scale from deviations above rounding noise.
  This recovery requires informative values in one-quarter of the columns.
  It also limits correction energy to 10% of input energy. Larger corrections
  could remove the sample instead of streaks.
- Fourier-SVD preserves prior output outside that edge case. Output comparisons
  covered the full measured CG-1D volume and noise-bearing synthetic cases.
  Of 540 measured sinograms, 533 follow the original path unchanged. The other
  seven have zero MAD in their column profiles. A median filter can produce
  exact zeros for most details on a smooth profile. No scale can be estimated
  from zero spread. Those seven inputs now return unchanged. That result is
  closer to the reference than the former standard-deviation fallback.
- The GUI can process logged attenuation sinograms in multiscale mode. Its
  Multiscale Streak panel now includes an "Input is already log-transformed
  (attenuation)" toggle. The toggle sets `log_domain_input`. Without it, the
  GUI applied a second logarithm to attenuation data. This made normalized
  CG-1D results unusable.
- Cancelling GUI processing returns the app to a ready state (#136). Previously,
  cancellation joined the worker and stopped reading its progress channel.
  The final cancellation message never reached the application state. The
  progress bar and Process button then remained stuck. Loading another file
  was the only recovery. Cancellation now drains the channel after the worker
  exits. A just-finished result still reaches the app. A missing worker result
  becomes an error. Otherwise, the app enters its existing cancelled state and
  enables Process again.
- Compare-view titles now center above their image panels (#137). The Original,
  Processed, and Difference labels previously clustered at the upper left. The
  layout reserved panel-wide cells but advanced only by each label's width.
- The GUI patch-size control now supports every value from 4 through 16 (#135).
  It replaces a menu limited to 4, 8, or 16. The core accepts every size in
  this range. Its optional Hadamard fast path applies only to 8-by-8 patches.
  Other sizes use the regular path. Patch size 7 is now selectable as the
  documentation recommends.
- Multiscale streak removal now follows Mäkinen et al. (2021) more closely. It
  denoises logarithmic attenuation data. It first averages adjacent sinogram
  rows, then processes overlapping horizontal segments. Each segment receives
  a local noise estimate. Each pyramid level receives the frequency profile of
  its remaining noise. Automatic estimation now measures streak amplitude.
  Previously, measurements from flat air regions could approach zero. This
  silently disabled the entire correction. Linear transmission input is logged
  and restored internally. Logged attenuation input requires
  `log_domain_input=True` or the matching GUI toggle. Otherwise, a second
  logarithm produces unusable output. Varying input without positive values is
  rejected with a message naming the flag. A constant blank slice remains
  unchanged. This rework affects only multiscale processing. Single-scale
  streak, generic, and Fourier-SVD processing otherwise retain prior behavior.

### Changed

- Import `compute_cdf` from `bm3dornl.utils` now (#138). The removed
  `bm3dornl.plot` module contained only this function. It computes an image's
  cumulative distribution function. The function itself is unchanged.
- Regenerate saved Fourier-SVD quality values for flat-background inputs. The
  fix above changes outputs that were previously almost identical to their
  inputs. On the benchmark phantom, the structural similarity index (SSIM)
  rises from 0.9510 to 0.9737. The unprocessed input also scored 0.9510.

### Documentation

- Install documentation now separates releases from clones. Clone builds,
  tests, and GUI runs use Pixi. `pip install -e .` is unsupported. The `[gui]`
  extra installs the released GUI binary from PyPI. Run a clone's GUI with
  `pixi run gui`. The previous contributor setup named a missing micromamba
  file. Its test commands also bypassed Pixi. Both now use supported Pixi tasks.
- `estimate_noise_sigma` documentation now describes vertical streak amplitude
  (#134). The former docstring described general image noise. Its example
  expected about 0.1 for independent Gaussian noise but produced about 0.013.
  Vertical smoothing intentionally suppresses independent pixel noise. The
  estimator also fills `sigma_random` when its value is at or below 1e-6, such as 0.0.
  Python docstrings,
  Rust comments, and the crate README now state this behavior. The example now
  constructs vertical streak noise. Tests cover both noise cases. Runtime
  behavior is unchanged.

### Known limitation

- Fourier-SVD can under-correct streaks above background noise. This occurs when
  noisy background occupies most columns, because MAD then measures background
  noise. Outputs in this case remain byte-identical to the previous release.
  A future fix needs reliable separation of sample and background columns.

### Internal

- Upgrade Pixi to 0.68 or newer before using this clone. `pixi.lock` now uses
  lock-file format 7. Other Neutron Data Project repositories use this format.
  The package URLs are unchanged from the previous lock. Only the file layout
  changed. Continuous integration now uses Pixi 0.78.0, which generated the
  lock.

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
