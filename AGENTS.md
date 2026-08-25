# AGENTS.md — bm3dornl

Shared instructions for AI coding agents working in this repo. **OpenAI Codex** and
other AGENTS.md-aware tools read this file directly (from the repo root); **Claude
Code** reads it via `.claude/CLAUDE.md`, which imports this file with `@../AGENTS.md`.
Keep this the single source of truth — edit instructions here, not in two places.

bm3dornl is BM3D streak/ring-artifact removal for neutron imaging: a Python API over a
Rust backend (PyO3 + maturin), plus a standalone egui GUI.

## Environment & commands (Pixi)

This is a **Pixi** project — run everything through `pixi run`. Do NOT use bare
`pip install` / `maturin` / `cargo` for the Python package outside the pixi env, and do
NOT add `sys.path` hacks (`bm3dornl` is installed editable).

| Task | Command |
|------|---------|
| Build the Rust extension (editable) | `pixi run build` (`maturin develop --release`) |
| All tests | `pixi run test` (→ `test-rust` + `test-python`) |
| Rust tests only | `pixi run test-rust` (`cargo test --workspace`) |
| Python tests only | `pixi run test-python` (`pytest`) |
| Lint | `pixi run lint` (→ `lint-fmt` + `lint-clippy`) |
| Run / debug the GUI | `pixi run gui` / `pixi run gui-debug` |
| Benchmarks | `pixi run bench` / `pixi run bench-rust` |
| Version | `pixi run version-show` / `version-{patch,minor,major}` |

## Pre-commit checklist (run before every commit)

```
pixi run lint     # cargo fmt --check + clippy -D warnings (warnings fail the build)
pixi run test     # cargo test --workspace + pytest
pixi run build    # confirm the extension still compiles
```

Fix all output before committing. Do not silence clippy with `#[allow(...)]`, and do not
skip tests.

## Project layout

- `src/bm3dornl/` — Python API. `bm3d.py` is the entry (`bm3d_ring_artifact_removal`);
  also `fourier_svd.py`, `phantom.py`, `plot.py`, `utils.py`. Thin orchestration over Rust.
- `src/rust_core/crates/`
  - `bm3d_core` — the algorithm (block matching, transforms, pipeline, multiscale, streak,
    fourier_svd, noise_estimation). Generic over f32/f64 via the `Bm3dFloat` trait.
    **Published to crates.io.**
  - `bm3d_python` — PyO3 bindings (the `bm3d_rust` module). Exposes **both** f32 and f64
    variants of every kernel.
  - `bm3d_gui_egui` — standalone egui desktop application.
- `tests/` — Python unit + integration tests.

2D inputs run entirely in Rust; 3D stacks are batch-orchestrated in Python over the Rust kernels.

## Conventions

- **f32/f64 parity**: `bm3d_core` is generic over `Bm3dFloat` and `bm3d_python` exposes both
  precisions. When you change a kernel or a binding, keep both paths in sync.
- **Defaults are shared**: the canonical BM3D defaults live in the Rust
  `Bm3dConfig`/`MultiscaleConfig` `Default` impls; the Python wrapper and GUI derive from /
  mirror them. Guard tests (`tests/test_defaults.py` and
  `orchestration::tests::test_default_config_matches_spec`) catch drift — update them when
  defaults change.
- **Commit attribution**: AI-assisted commits end with an `Assisted-With:` trailer
  (e.g. `Assisted-With: Claude <model> <noreply@anthropic.com>`), NOT `Co-Authored-By:`.
- **Don't touch `.claude/worktrees/`** — managed by Claude Code for isolated sessions.

## Branches & review

- Promotion: `next` (default/dev branch) → `qa` → `main`. All three require PR approval;
  admins are exempt (`enforce_admins=false`), which is how releases are cut directly.
- Review workflow: changes are often implemented by **Codex** and **gated by Claude**.
  Gating = review the full diff against the surrounding code, verify correctness + f32/f64
  parity + the Python↔Rust boundary, run `pixi run lint`/`test`/`build`, and give a clear
  verdict (sound / needs changes / blocked) before merge — not a rubber stamp.

## Releases

Claude Code users: see the `release` skill (`.claude/skills/release/`). In short:
`pixi run version-{minor,patch}` → refresh `Cargo.lock` (a build does it; `version.py` does
not) → roll `CHANGELOG.md` → validate → commit → promote `next → qa → main` → tag `vX.Y.Z`,
which triggers PyPI + crates.io + GitHub release via `.github/workflows/release.yml`.
Versioning is pre-1.0 semver: behavior/feature changes → **minor**, backward-compatible
fixes → **patch**.
