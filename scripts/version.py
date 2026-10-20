#!/usr/bin/env python3
"""Version management script for bm3dornl.

Single source of truth: src/rust_core/Cargo.toml [workspace.package].version
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

CARGO_WORKSPACE = REPO_ROOT / "src" / "rust_core" / "Cargo.toml"

PYTHON_MAIN = REPO_ROOT / "pyproject.toml"
PYTHON_GUI = (
    REPO_ROOT / "src" / "rust_core" / "crates" / "bm3d_gui_egui" / "pyproject.toml"
)


def read_version() -> str:
    content = CARGO_WORKSPACE.read_text()
    match = re.search(r'\[workspace\.package\]\s*\n\s*version\s*=\s*"([^"]+)"', content)
    if not match:
        raise ValueError(
            f"Could not find [workspace.package] version in {CARGO_WORKSPACE}"
        )
    return match.group(1)


def write_cargo_version(version: str) -> None:
    content = CARGO_WORKSPACE.read_text()
    new_content = re.sub(
        r'(\[workspace\.package\]\s*\n\s*version\s*=\s*)"[^"]+"',
        f'\\1"{version}"',
        content,
    )
    CARGO_WORKSPACE.write_text(new_content)
    print(f"  Updated {CARGO_WORKSPACE.relative_to(REPO_ROOT)}")


def sync_python_main(version: str) -> None:
    content = PYTHON_MAIN.read_text()

    new_content = re.sub(
        r'^(version\s*=\s*)"[^"]+"',
        f'\\1"{version}"',
        content,
        flags=re.MULTILINE,
    )

    new_content = re.sub(
        r'(gui\s*=\s*\["bm3dornl-gui)==[^"]+("\])',
        f"\\1=={version}\\2",
        new_content,
    )

    PYTHON_MAIN.write_text(new_content)
    print(f"  Updated {PYTHON_MAIN.relative_to(REPO_ROOT)}")


def sync_python_gui(version: str) -> None:
    content = PYTHON_GUI.read_text()
    new_content = re.sub(
        r'^(version\s*=\s*)"[^"]+"',
        f'\\1"{version}"',
        content,
        flags=re.MULTILINE,
    )
    PYTHON_GUI.write_text(new_content)
    print(f"  Updated {PYTHON_GUI.relative_to(REPO_ROOT)}")


def sync_all(version: str) -> None:
    print(f"Syncing version {version} to Python files...")
    sync_python_main(version)
    sync_python_gui(version)
    print("Done!")


def bump_version(version: str, component: str) -> str:
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {version} (expected X.Y.Z)")

    major, minor, patch = map(int, parts)

    if component == "major":
        return f"{major + 1}.0.0"
    elif component == "minor":
        return f"{major}.{minor + 1}.0"
    elif component == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Unknown component: {component}")


def cmd_show() -> None:
    version = read_version()
    print(f"Current version: {version}")


def cmd_sync() -> None:
    version = read_version()
    sync_all(version)


def cmd_bump(component: str) -> None:
    old_version = read_version()
    new_version = bump_version(old_version, component)

    print(f"Bumping {component} version: {old_version} -> {new_version}")
    print()

    print("Updating Cargo workspace...")
    write_cargo_version(new_version)
    print()

    sync_all(new_version)
    print()
    print(f"Version bumped to {new_version}")
    print()
    print("Next steps:")
    print("  1. Review changes: git diff")
    print("  2. Stage files: git add pyproject.toml src/rust_core/Cargo.toml \\")
    print("                       src/rust_core/crates/bm3d_gui_egui/pyproject.toml")
    print(f'  3. Commit: git commit -m "chore: bump version to {new_version}"')


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    command = sys.argv[1].lower()

    if command == "show":
        cmd_show()
    elif command == "sync":
        cmd_sync()
    elif command in ("patch", "minor", "major"):
        cmd_bump(command)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
