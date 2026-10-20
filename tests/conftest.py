from pathlib import Path
import pytest
from shutil import rmtree
from tempfile import mkdtemp


@pytest.fixture(scope="function")
def tmpdir():
    r"""Create directory, then delete the directory and its contents upon test exit"""
    try:
        temporary_dir = Path(mkdtemp())
        yield temporary_dir
    finally:
        rmtree(temporary_dir)
