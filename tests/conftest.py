# standard imports
from pathlib import Path
import pytest
from shutil import rmtree
from tempfile import mkdtemp


# NOTE: pytest fixtures tmp_path and tmp_path_factory are NOT deleting the temporary directory, hence this fixture
@pytest.fixture(scope="function")
def tmpdir():
    r"""Create directory, then delete the directory and its contents upon test exit"""
    try:
        temporary_dir = Path(mkdtemp())
        yield temporary_dir
    finally:
        rmtree(temporary_dir)
