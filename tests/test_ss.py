import re
from importlib import metadata

import pytest

import ss


def test_version() -> None:
    try:
        # Check if the version is a string
        assert isinstance(ss.__version__, str)

        # Check if the version string is in the correct format
        assert re.match(r"\d+\.\d+\.\d+", ss.__version__)
    except metadata.PackageNotFoundError as e:
        pytest.fail(f"Package not installed: {e}")
