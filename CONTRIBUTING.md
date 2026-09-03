# Contributing to BM3D-ORNL

Thank you for contributing to BM3D-ORNL. This guide explains the project workflow.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Contact](#contact)

## Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. **Fork the repository.** Create a fork of the
   [bm3dornl repository](https://github.com/ornlneutronimaging/bm3dornl).

2. **Clone your fork.**

    ```bash
    git clone https://github.com/your-username/bm3dornl.git
    cd bm3dornl
    ```

3. **Add the upstream remote.**

    ```bash
    git remote add upstream https://github.com/ornlneutronimaging/bm3dornl.git
    ```

4. **Use Pixi.** Contributors must use [Pixi](https://prefix.dev) for clone
   builds and tests. `pip install -e .` is unsupported: Pixi pins every
   dependency in `pixi.lock`, and pip resolves its own set. Pixi does not
   supply Rust; install a stable Rust toolchain with rustup first.

   After cloning, run `pixi run build`, `pixi run test`, then `pixi run gui`.

    ```bash
    pixi run build     # build the Rust extension
    pixi run test      # run the Rust and Python tests
    pixi run gui       # run the clone's GUI
    ```

   Each `pixi run` command installs the environment on demand. Run
   `pixi install` first only to create the environment separately.

   See the README's [How to install](README.md#how-to-install) section for the
   install policy. The [Development](README.md#development) section lists all
   supported tasks.

## Development Workflow

- **Create a branch.** Use a focused name for your change.

    ```bash
    git switch -c feature/your-feature-name
    ```

- **Make your changes.** Keep each change focused.

    ```bash
    pixi run lint
    ```

  Run the lint task before committing. It checks Rust formatting and Clippy.

- **Write tests.** Cover new behavior and regressions. See [Testing](#testing).

- **Commit the change.** Use a concise, meaningful message.

    ```bash
    git add .
    git commit -m "Description of your changes"
    ```

- **Push the branch.** Send it to your fork.

    ```bash
    git push origin feature/your-feature-name
    ```

- **Open a pull request.** Target the original repository's `next` branch.
  Describe the change and its verification.

## Coding Standards

- **Python style:** Follow PEP 8.
- **Docstrings:** Use NumPy-style docstrings for public modules, classes, and functions.
- **Type annotations:** Annotate function signatures.
- **Imports:** Group standard-library, third-party, and local imports. Use absolute imports.
- **Rust:** Run `pixi run lint`. It checks formatting and treats Clippy warnings as errors.

## Testing

Cover your changes with tests. Rust tests live under `src/rust_core`.
Python tests live under `tests/`.

- **Run all tests.** This task runs Rust and Python tests.

    ```bash
    pixi run test
    ```

- **Run one test suite.**

    ```bash
    pixi run test-rust
    pixi run test-python
    ```

## Submitting Changes

1. **Run the checks.** Ensure `pixi run lint`, `pixi run test`, and
   `pixi run build` pass.

2. **Update documentation.** Revise any affected pages.

3. **Open a pull request.** Describe the change and reference related issues.

4. **Address review feedback.** Maintainers may request changes.

## Reporting Issues

Report bugs and feature requests on the
[GitHub issues page](https://github.com/ornlneutronimaging/bm3dornl/issues).
Include reproduction steps for bugs.

Start a [GitHub Discussion](https://github.com/ornlneutronimaging/bm3dornl/discussions)
for usage questions. This keeps issues focused on defects and features.

## Contact

Contact the [repository maintainer](mailto:zhangc@ornl.gov) for further help.
