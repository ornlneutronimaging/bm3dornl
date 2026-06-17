---
name: release
description: Cut and publish a bm3dornl release — version bump, changelog, validate, promote next→qa→main, tag, and verify the publish landed
user-invokable: true
---

# Release Pipeline (bm3dornl)

Cut a new bm3dornl release. The version bump + tag is local git work; building and
publishing (PyPI + crates.io + GitHub release) happen on CI when the `vX.Y.Z` tag
arrives — `.github/workflows/release.yml` triggers on `tags: ['v*']`.

## Arguments
- No args: decide the version (advise patch vs minor per the rule below) and confirm with the user.
- `<X.Y.Z>`: target version, no leading `v`.

## Versioning (pre-1.0 semver)
bm3dornl is `0.x`. Map the release scope to:
- **minor** (`0.9.0 → 0.10.0`): new features OR any behavior/default change that is
  not backward-compatible (e.g. changing a default `mode`, `threshold`, or sigma).
- **patch** (`0.9.0 → 0.9.1`): backward-compatible bug fixes only.

## Step 1 — Pre-flight
1. Review the release scope: `git log $(git describe --tags --abbrev=0)..origin/next --oneline`.
2. Branch off the up-to-date `next`, clean tree:
   `git fetch origin && git switch -c release/X.Y.Z origin/next`.
3. Confirm the tag is free: `git ls-remote --tags origin vX.Y.Z` must be empty.

## Step 2 — Bump the version
`pixi run version-{minor|patch|major}`. Source of truth is
`src/rust_core/Cargo.toml [workspace.package].version`; the script also syncs
`pyproject.toml` (`[project].version` + the `bm3dornl-gui==X.Y.Z` optional-dep pin) and
`src/rust_core/crates/bm3d_gui_egui/pyproject.toml`. All three crates use
`version.workspace = true`, so the bump propagates. Verify with `pixi run version-show`.

## Step 3 — Refresh Cargo.lock (GOTCHA)
`version.py` does NOT touch `src/rust_core/Cargo.lock` (or `pixi.lock`). Run
`pixi run build` so the three workspace crates show the new version in `Cargo.lock`, and
include the lockfile in the release commit. Leave `pixi.lock` out — its drift is environmental.

## Step 4 — CHANGELOG
Roll `## Unreleased` in `CHANGELOG.md` into a dated `## X.Y.Z - YYYY-MM-DD` section.
Build the notes from the actual diffs (`git show <sha>`), not just PR titles; group as
Changed / Fixed / Security / Internal. Flag any behavior/default change prominently.

## Step 5 — Validate (all must pass before tagging)
```
pixi run lint      # fmt + clippy -D warnings
pixi run build     # maturin --release
pixi run test      # cargo test --workspace + pytest
```

## Step 6 — Commit
```
git add CHANGELOG.md pyproject.toml src/rust_core/Cargo.toml \
        src/rust_core/crates/bm3d_gui_egui/pyproject.toml src/rust_core/Cargo.lock
git commit -m "chore: bump version to X.Y.Z

<one-line release summary>

Assisted-With: Claude <model> <noreply@anthropic.com>"
```

## Step 7 — Promote next → qa → main
All three branches require a PR + 1 approval, but `enforce_admins=false`, so an admin
pushes directly (a self-opened PR cannot be self-approved). `qa`/`main` trail `next` and
fast-forward cleanly:
```
git push origin HEAD:next      # ff next to the release commit
git push origin HEAD:qa        # ff qa
git push origin HEAD:main      # ff main
```
Verify: `for b in next qa main; do git ls-remote origin refs/heads/$b; done` — all three
should point at the release commit.

## Step 8 — Tag (the publish trigger — IRREVERSIBLE)
```
git tag -a vX.Y.Z <release-commit> -m "vX.Y.Z"
git push origin vX.Y.Z
```
This fires `release.yml`: builds wheels → publishes `bm3dornl` to PyPI, `bm3d_core` to
crates.io, and creates the GitHub release (~4–5 min). **PyPI and crates.io do not allow
reusing a version** — make sure everything is right before pushing the tag.

## Step 9 — Monitor
```
gh run watch $(gh run list --workflow=release.yml --limit 1 --json databaseId --jq '.[0].databaseId') --exit-status
```

## Step 10 — Verify artifacts landed (report each explicitly)
```
gh release view vX.Y.Z                                                  # GitHub release
curl -sf https://pypi.org/pypi/bm3dornl/X.Y.Z/json >/dev/null && echo PyPI-OK || echo PyPI-MISSING
curl -s -A "bm3dornl-release-check" https://crates.io/api/v1/crates/bm3d_core \
  | python3 -c "import sys,json; print('crates.io newest:', json.load(sys.stdin)['crate']['newest_version'])"
```
Note: the crates.io API **requires a `User-Agent` header** — omitting it returns an error.

## Failure modes
- **Partial publish**: a version can't be re-uploaded to PyPI/crates.io. If a job fails
  *after* a successful publish of the same version, bump to the next version and
  re-release. crates.io publish skips already-published crates, so re-running the workflow
  on the same tag is safe for the crate side.
- **Tag pushed in error** (before CI published): `git push --delete origin vX.Y.Z` +
  `git tag -d vX.Y.Z`, fix, re-tag.
- **Workflow didn't start**: confirm the tag matches `v*` and check the Actions tab.
