# Contributing to BM3D-ORNL

Thank you for considering contributing to the BM3D-ORNL project!
We welcome contributions from the community and are grateful for your help in improving this library.
This guide provides instructions on how to contribute to the project.

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

1. **Fork the Repository**: Fork the [bm3dornl repository](https://github.com/ornlneutronimaging/bm3dornl) to your GitHub account.

2. **Clone the Fork**: Clone your forked repository to your local machine.

    ```bash
    git clone https://github.com/your-username/bm3dornl.git
    cd bm3dornl
    ```

3. **Set Upstream Remote**: Add the original repository as an upstream remote.

    ```bash
    git remote add upstream https://github.com/ornlneutronimaging/bm3dornl.git
    ```

4. **Create a Virtual Environment**: Set up a virtual environment to manage dependencies.

    ```bash
    micromamba create -f environment.yml
    micromamba activate bm3dornl
    ```

## Development Workflow

- **Create a Branch**: Create a new branch for your feature or bugfix.

    ```bash
    git checkout -b feature/your-feature-name
    ```

- **Make Changes**: Make your changes in the codebase. Use `pre-commit` to help you format your code and check for common issues.

    ```bash
    pre-commit install
    ```

> Note: you only need to run `pre-commit install` once. After that, the pre-commit checks will run automatically before each commit.

- **Write Tests**: Write tests for your changes to ensure they are well-tested. See the [testing](#testing) section for more details.

- **Commit Changes**: Commit your changes with a meaningful commit message.

    ```bash
    git add .
    git commit -m "Description of your changes"
    ```

- **Push Changes**: Push your changes to your forked repository.

    ```bash
    git push origin feature/your-feature-name
    ```

- **Open a Pull Request**: Open a pull request (PR) from your forked repository to the `next` branch of the original repository. Provide a clear description of your changes and any relevant information.

## Coding Standards

- **PEP 8**: Follow the PEP 8 style guide for Python code.
- **Docstrings**: Use `numpy` docstrings style to document all public modules, classes, and functions.
- **Type Annotations**: Use type annotations for function signatures.
- **Imports**: Group imports into standard library, third-party, and local module sections. Use absolute imports.

## Testing

We use `pytest` for testing. Ensure that your changes are covered by tests.

- **Run Tests**: Run the tests using `pytest`.

    ```bash
    pytest -v
    ```

- **Check Coverage**: Check the test coverage.

    ```bash
    pytest --cov=src/bm3dornl
    ```

## Submitting Changes

1. **Ensure Tests Pass**: Make sure all tests pass and the coverage is satisfactory.

2. **Update Documentation**: If your changes affect the documentation, update the relevant sections.

3. **Open a Pull Request**: Open a pull request with a clear description of your changes. Reference any related issues in your PR description.

4. **Review Process**: Your pull request will be reviewed by the maintainers. Be prepared to make changes based on feedback.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on the [GitHub issues page](https://github.com/ornlneutronimaging/bm3dornl/issues).
Provide as much detail as possible, including steps to reproduce the issue if applicable.

## Contact

If you have any questions or need further assistance, please contact the [repo maintainer](zhangc@ornl.gov).

Thank you for contributing to BM3D-ORNL!
