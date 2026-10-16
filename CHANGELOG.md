# Changelog

## 0.12.0 - 2026-10-15

### Changed

- **Behavior change:** generic mode, the general-purpose BM3D denoiser, now
  tests every candidate position in the block-matching search window, as BM3D
  specifies (#163). The search used to step by `step_size`, the stride
  between reference patches. That left more bias and blocky errors at edges
  that do not sit on the reference grid. Generic output changes. Generic
  processing takes 2 to 3 times as long at the default `search_window` (a
  512 x 512 image: 0.56 s instead of 0.27 s on one thread), and more with a
  larger window (about 5 to 6 times at `search_window=64`).
- New `search_step` parameter: the stride between candidate positions in the
  block-matching search (#163). `None`, the default, uses 1 in generic mode
  and `step_size` in streak, multiscale and Fourier-SVD modes, so their
  search does not change. Pass `search_step=step_size` to restore the old
  generic output. `search_step=0` raises a `ValueError`.
- `bm3d_ring_artifact_removal` takes `search_step` as its last parameter. The
  `bm3d_rust` bindings take it as a last keyword argument, so existing
  positional calls keep working. In the low-level kernel bindings
  (`bm3d_hard_thresholding`, `bm3d_wiener_filtering` and their stack and f64
  variants), `None` means `step_size`.
- Rust core: `Bm3dConfig` has a new field `search_step: Option<usize>` and
  `Bm3dKernelConfig` a new field `search_step: usize`. Struct literals of
  `Bm3dKernelConfig`, and of `Bm3dConfig` without `..Default::default()`, must
  set it.
- **Behavior change:** a `step_size` larger than `patch_size` is now refused
  with `ValueError: step_size (s) must not exceed patch_size (p)` on every
  path (#166). Such a stride left gaps between the reference patches, and the
  pixels in the gaps came back unfiltered: in generic mode on a 64 x 80 image
  at `patch_size=8`, `step_size=12` left 2880 of 5120 pixels unchanged.
  `step_size` equal to `patch_size` stays valid. In the Rust core,
  `Bm3dConfig::validate`, `run_bm3d_step` and `run_bm3d_step_stack` return
  the same error. The two kernels also return an error, instead of a panic,
  for a `step_size` or `search_step` of 0.

### Fixed

- The bottom and right edges of the image are now filtered in generic and
  streak modes (2D and 3D input) and in Fourier-SVD mode (2D input) (#163).
  Reference patches sat only at multiples of `step_size`. When the
  image size minus `patch_size` was not a multiple of `step_size` (at the
  defaults, any size not divisible by 4), the last rows or columns came back
  unfiltered, with all of their noise. For example, a 720 x 725 sinogram lost
  its last column. One extra reference row and column now reach each edge.
  Where the grid already reaches both edges, streak and Fourier-SVD output,
  and generic output with `search_step=step_size`, stay bit-identical.
- Multiscale output can also change, because its internal scales often have
  sizes off the reference grid. The change is usually small (at most 6.1e-3
  in our tests). On some inputs, though, it
  changes the decision of the multiscale wall gate, which keeps straight
  object edges from being removed as stripes. In a synthetic test with
  `sigma_random` set to the true noise level, the gate then missed a straight
  object edge more often, and that edge came out smeared (RMSE 0.0105 to
  0.0123). With the default `sigma_random`, 0.11.0 already misses this edge.
  The gate is tracked in #167.
- `mode` in `bm3d_ring_artifact_removal` is now case-insensitive on every
  path, as it already was for 2D single-scale input. In 3D, `mode="Streak"`
  used to run generic processing, and `multiscale=True` refused `"Streak"`. A
  `mode` that is not a string now raises `TypeError` on every path.
- An image smaller than `patch_size` in either dimension now raises
  `ValueError: Image size (rows, cols) is smaller than patch_size p` on every
  path, as 2D input already did (#165). The 3D stack path, the low-level
  kernel bindings and the `test_block_matching_rust` debug bindings used to
  panic with `pyo3_runtime.PanicException`, which `except Exception` does not
  catch. Multiscale processing used to return an image with fewer rows than
  `patch_size` unfiltered. The Rust kernels `run_bm3d_step` and
  `run_bm3d_step_stack` return the same error, from the new public check
  `bm3d_core::pipeline::check_image_size`.
- Input arrays that are not C-contiguous now give the same output as C-order
  input (#164). The Rust fast paths read pixel data in memory order but
  indexed it as row-major, so a Fortran-order or negative-stride array was
  read with rows and columns, or the column order, mixed up. This affected 3D
  stacks with F-contiguous slices (for example a `transpose(0, 2, 1)` view, or
  `np.asfortranarray` of a one-slice stack: RMSE 0.352 instead of 0.006
  against the clean image), the `bm3d_rust` bindings, `fourier_svd_removal`
  and `estimate_noise_sigma`. 2D input to `bm3d_ring_artifact_removal` was
  not affected. The bindings now convert input to C order, so every layout
  gives the C-order output exactly. In the Rust core, the fast paths take
  only standard-layout arrays; other layouts take the indexed path, which
  agrees with the standard-layout result to rounding.

### Documentation

- The README DOI badge now points at the Zenodo concept DOI (#162).

### Internal

- CI action updates raised `softprops/action-gh-release` from 3.0.2 to 3.0.3,
  `prefix-dev/setup-pixi` from 0.10.2 to 0.11.0, `actions/download-artifact`
  from 8.0.1 to 8.0.2 and `actions/upload-artifact` from 7.0.1 to 7.0.2
  (#161, #170, #171, #172).

## 0.11.0 - 2026-09-03

### Fixed

- GUI HDF5 volumes now load about 16 times faster on the 591-slice gzip test:
  about 4 seconds instead of 62. The progress reader added after 0.10.0 had read
  slice by slice, repeatedly decompressing chunks. Chunk-aligned reads now cover
  every chunked dataset, decompress each chunk once, and preserve visible
  progress.
- Loaded HDF5 values remain identical to the previous reader. A test covers a
  short final block.
- Optional notebook data now clones anonymously over HTTPS, so recursive clones
  no longer require an ORNL account or registered SSH key (#132). The README
  explains that tests and builds do not need this 1.4 GB Git LFS submodule.
- Fresh test-data submodule checkouts now succeed. The pointer now uses the
  correct full commit hash instead of a mistyped, nonexistent one.
- Fourier-SVD now removes streaks from flat-background sinograms through
  `fourier_svd_removal` and `FourierSvd` mode (#133). Rescue starts when the old
  threshold falls below 1e-6 of the largest deviation. Scale is then estimated
  from deviations above rounding noise.
- Fourier-SVD controls now affect flat-background output. Its magnitude gate
  removes detail below its threshold and protects detail above it. Flat air
  drove its median absolute deviation, the median distance from the median, to
  about 1e-11, protecting all detail and disabling both `fft_alpha` and
  `notch_width`.
- Unsafe Fourier-SVD rescues now return the input unchanged. Rescue requires
  informative values in at least one-quarter of columns and never fewer than
  16 columns. It declines when correction would exceed 10% of input energy;
  the correction is not clamped because it could remove the sample.
- Outside this degenerate case, Fourier-SVD output remains byte-identical to
  0.10.0. Output-for-output checks covered the full measured CG-1D volume and
  noise-bearing synthetic cases. Of 540 measured sinograms, 533 remain
  byte-identical.
- Seven CG-1D sinograms with zero median absolute deviation now return
  unchanged. This is closer to the reference than the former standard-deviation
  fallback. A median filter can make most details exactly zero on a smooth
  profile, leaving no scale to estimate.
- The GUI now processes logged attenuation sinograms correctly in multiscale
  mode. The Multiscale Streak panel adds an "Input is already log-transformed
  (attenuation)" toggle that sets `log_domain_input`. Without it, normalized
  CG-1D data received a second logarithm and became unusable.
- Canceling GUI processing now returns the app to a ready state (#136).
  Previously, cancellation joined the worker and stopped reading its progress
  channel. The lost final message left progress and Process stuck until another
  file loaded.
- Cancellation now drains messages after the worker exits. A just-finished
  result survives, while normal cancellation reaches the existing canceled
  state and enables Process. A worker panic without a result becomes an error;
  a clean exit without a final result becomes canceled.
- Compare-view titles now center above their image panels (#137). Original,
  Processed, and Difference previously clustered at the upper left. The layout
  reserved panel-wide cells but advanced only by each label's width.
- GUI patch sizes now include every integer from 4 through 16 (#135). The old
  menu offered only 4, 8, and 16; size 7 is now selectable as documented. All
  sizes use the core's regular path; only 8-by-8 patches offer the optional
  Hadamard fast transform.
- Multiscale streak removal now follows Mäkinen et al. (2021) more closely. It
  denoises logarithmic attenuation data. For taller inputs, it first averages
  adjacent sinogram rows.
- Multiscale denoising now processes overlapping horizontal segments. Each
  segment gets a local noise estimate. Each processing scale gets the frequency
  profile of its remaining noise.
- Automatic multiscale noise estimation now measures streak amplitude. Flat air
  previously drove estimates near zero. This silently disabled all correction.
- Multiscale processing now expects linear transmission unless
  `log_domain_input=True` or the matching GUI toggle is set. It logs and restores
  linear data; unflagged attenuation data is logged twice and becomes unusable.
  Varying data without positive values is rejected with a message naming the
  flag, while a constant blank slice passes through unchanged.
- Single-scale streak and generic processing remain unchanged by the multiscale
  rework. Fourier-SVD also remains unchanged by that rework. The Fourier-SVD
  fix above is separate.

### Changed

- The GUI now accepts a volume or TIFF sequence on its command line.
  `-d` or `--dataset` selects an HDF5 dataset; a file containing one 3D dataset
  opens it automatically. Other HDF5 files open the dataset browser, while TIFF
  stacks and sequences load directly.
- Every volume loads on a background thread, preventing interface freezes.
  Known HDF5 slice totals show exact progress; unknown totals use an
  animated bar.
- `--called-from-app FILE` adds a return button that saves the processed volume
  as HDF5 dataset `/data`, then closes the GUI. This lets a launching
  application round-trip data through the GUI.
- The GUI now explains the selected algorithm in a collapsible "About this
  method" panel (#156). It covers Generic BM3D, Streak, Multiscale Streak, and
  Fourier-SVD.
- Each method view explains how and when to use it, then links its GitHub source
  and related literature (#156). References cover Dabov (2007), Mäkinen (2020,
  2021), Münch (2009), and Vo (2018).
- The MIT license now credits 2024-2026 UT-Battelle, LLC (Oak Ridge National
  Laboratory), replacing "Neutron Scattering Software."
- `bm3dornl.plot`, which contained only `compute_cdf`, has been removed (#138).
  The old import now fails; use
  `from bm3dornl.utils import compute_cdf`. The function still computes an
  image's cumulative distribution and is otherwise unchanged.
- Saved Fourier-SVD quality values now reflect corrected flat-background
  output. These outputs were previously almost identical to their inputs. On
  the benchmark phantom, structural similarity rises from 0.9510 to 0.9737;
  the unprocessed input scores 0.9510.

### Security

- Continuous integration and release workflows now reduce supply-chain risk by
  pinning every GitHub Action to a full commit SHA (#118, #121). Pins retain
  version comments, and Dependabot maintains them.
- Pure continuous-integration workflows now grant only read access to repository
  contents (#118). Rust toolchain steps explicitly request stable, preserving
  that channel after pinning. The pinning itself changed no action versions or
  workflow behavior.

### Documentation

- Installation instructions now separate releases from source clones. The
  `[gui]` extra installs the released GUI binary from PyPI. For clones, use Pixi
  for builds, tests, and the GUI; `pip install -e .` is unsupported, and
  `pixi run gui` starts the GUI.
- Contributor setup now uses supported Pixi tasks. The previous instructions
  named a missing micromamba file. They also ran tests outside Pixi.
- `estimate_noise_sigma` documentation now describes vertical streak amplitude
  instead of general image noise (#134). Independent Gaussian noise produced
  about 0.013, not the former example's expected 0.1, because vertical smoothing
  suppresses independent pixel noise. The example now uses vertical streak
  noise, and tests cover both noise cases.
- Noise-estimation documentation now states that `sigma_random <= 1e-6`,
  including 0.0, enables automatic estimation. Python docstrings, Rust comments,
  and the crate README now describe this behavior. Runtime behavior is
  unchanged.
- Optional test-data documentation now records the measured data's provenance
  (#140). `tomostack_small.h5` comes from CG-1D acquisition IPTS-30610, raw file
  `2023_08_11_1D_redo`.
- `clean_tomostack` is the same acquisition denoised by the closed-source
  reference, not ground truth or a second measurement (#140). `sino.npy` is a
  simulated Shepp-Logan sinogram.
- Every notebook now names its extra prerequisites (#139, #144). The tutorial
  adds `matplotlib`; real-data notebooks add `matplotlib`, `h5py`, and commands
  to fetch optional test data.
- The real-data notebooks also note anonymous HTTPS access and the approximately
  1.4 GB Git LFS download (#139, #144).
- The README now links the full documentation and names its installation,
  tutorial, parameters, GUI, and API sections (#141, #144). It also links the
  contributing instructions and Code of Conduct.
- Support guidance now directs defects and requests to Issues and usage questions
  to Discussions (#141, #144). The maintainer email link now works.
- The API reference now renders docstrings for all 10 public functions (#143,
  #144). Its environment now installs runtime dependencies and continues mocking
  the compiled extension. Previously, five imports failed and the page contained
  only headers.
- The Fourier-SVD formula is now valid reStructuredText, leaving the documentation
  build warning-free (#143, #144). The GUI guide now says the scroll wheel zooms
  and the slice slider navigates stacks.

### Known limitation

- Fourier-SVD can still under-correct streaks stronger than background noise
  when noisy background fills most columns. The spread estimate then measures
  background noise. These outputs remain byte-identical to 0.10.0; a future fix
  needs reliable separation of sample and background columns.

### Internal

- Source clones now require Pixi 0.68 or newer because `pixi.lock` uses format
  7. The lock preserves prior package URLs and changes only file layout, matching
  other Neutron Data Project repositories. Continuous integration moved from
  Pixi 0.62.2 to 0.78.0, which generated the lock.
- GUI maintenance keeps builds clean on newer stable Rust versions (#122, #127).
  Nine ambiguous floats now use explicit `f32` literals, avoiding future compiler
  errors. The command-line loading code now follows rustfmt and Clippy without
  behavior changes.
- CI action updates raised `actions/checkout` from 6 to 7.0.1,
  `prefix-dev/setup-pixi` from 0.9.6 to 0.10.2, and
  `softprops/action-gh-release` from 3.0.1 to 3.0.2. They also raised
  `pypa/gh-action-pypi-publish` from 1.14.0 to 1.14.2 and refreshed
  `Swatinem/rust-cache` v2 (#119, #120, #123, #124, #125, #128, #129, #130,
  #151).

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
