# python_project_template
This repository is a template repository for Python projects under neutrons.
After you create a new repository using this repo as template, please follow the following steps to adjust it for the new project.

## Codebase Adjustments

1. Adjust the branch protection rules for the new repo. By default, we should protect the `main` (stable), `qa` (release candidate), and `next` (development) branches.

    1.1 Go to the `Settings` tab of the new repo.

    1.2 Click on `Branches` on the left side.

    1.3 Click on `Add rule` button.

    1.4 Follow the instructions from Github.


2. Change the License if MIT license is not suitable for you project. For more information about licenses, please
refer to [Choose an open source license](https://choosealicense.com/).


3. Update the environment dependency file `environment.yml`, which contain both runtime and development dependencies.
For more information about conda environment file, please refer to [Conda environment file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-file-manually).

    3.1 Specify environment 'name' field to match package name

    3.2 We strongly recommended using a single `environment.yml` file to manage all the dependencies,
including the runtime and development dependencies.

    3.3 Please add comments to the `environment.yml` file to explain the dependencies.

    3.4 Please prune the dependencies to the minimum when possible,
we would like the solver to figure out the dependency tree for us.


4. Adjust pre-commit configuration file, `.pre-commit-config.yaml` to enable/disable the hooks you need.
For more information about pre-commit, please refer to [pre-commit](https://pre-commit.com/).


5. Having code coverage, `codecov.yaml` is **strongly recommended**,
please refer to [Code coverage](https://coverage.readthedocs.io/en/coverage-5.5/) for more information.


6. Adjust the demo Github action yaml files for CI/CD. For more information about Github action,
please refer to [Github action](https://docs.github.com/en/actions).

    6.1 Specify package name at: .github/workflows/package.yml#L34

    6.2 Specify package name at: .github/workflows/package.yml#L46


7. Adjust the conda recipe, `conda-recipe/meta.yaml` to provide the meta information for the conda package.
For more information about conda recipe, please refer to [Conda build](https://docs.conda.io/projects/conda-build/en/latest/).

    7.1 Specify package name at: conda.recipe/meta.yaml#L15

    7.2 Update license family, if necessary: conda.recipe/meta.yaml#L42


8. Adjust `pyproject.toml` to match your project. For more information about `pyproject.toml`,
please refer to [pyproject.toml](https://www.python.org/dev/peps/pep-0518/).

    8.1 Specify package name at: pyproject.toml#L2

    8.2 Specify package description at: pyproject.toml#L3

    8.3 Specify package name at: pyproject.toml#L39

    8.4 Specify any terminal entry points (terminal commands) at : pyproject.toml#48.
In the example, invoking `packagename-cli` in a terminal is equivalent to running the python script
`from packagenamepy.packagename.import main; main()"

    8.5 Projects will use a  single `pyproject.toml` file to manage all the project metadata,
including the project name, version, author, license, etc.

    8.6 Python has moved away from `setup.cfg`/`setup.py`, and we would like to follow the trend for our new projects.


10. Specify package name at  src/packagenamepy


11. Specify package name at: src/packagenamepy/packagename.py

12. If a GUI isn't used, delete the MVP structure at src/packagenamepy:
    11.1: mainwindow.py
    11.2: home/
    11.3: help/


11. Clear the content of this file and add your own README.md as the project README file.
We recommend putting badges of the project status at the top of the README file.
For more information about badges, please refer to [shields.io](https://shields.io/).

## Repository Adjustments

### Add an access token to anaconda

Here we assume your intent is to upload the conda package to the [anaconda.org/neutrons](https://anaconda.org/neutrons) organization.
An administrator of _anaconda.org/neutrons_ must create an access token for your repository in the [access settings](https://anaconda.org/neutrons/settings/access).

After created, the token must be stored in a _repository secret_:
1. Navigate to the main page of the repository on GitHub.com.
2. Click on the "Settings" tab.
3. In the left sidebar, navigate to the "Security" section and select "Secrets and variables" followed by "Actions".
4. Click on the "New repository secret" button.
5. Enter `ANACONDA_TOKEN` for the secret name
6. Paste the Anaconda access token
7. Click on the "Add secret" button
8. Test the setup by creating a release candidate tag,
which will result in a package built and uploaded to https://anaconda.org/neutrons/mypackagename

### Add an access token to codecov
Follow the instructions in the [Confluence page](https://ornl-neutrons.atlassian.net/wiki/spaces/NDPD/pages/103546883/Coverage+reports)
to create the access token.
