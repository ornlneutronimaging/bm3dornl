# Changelog

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
