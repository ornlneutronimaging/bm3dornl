# Changelog

## Unreleased

- Changed the public Python default for `bm3d_ring_artifact_removal` from `mode="generic"` to `mode="streak"` so the API default matches the package's ring-artifact removal focus and the GUI/default documentation.
- Changed the public Python `threshold` default to `None` so single-scale processing uses the Rust default `2.7` and multi-scale processing uses the Rust multi-scale default `3.5` unless callers provide an explicit threshold.
