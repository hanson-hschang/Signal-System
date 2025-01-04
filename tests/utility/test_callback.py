from pathlib import Path

import numpy as np
import pytest

from ss.utility.callback import Callback


class TestCallback:

    @pytest.fixture
    def callback(self) -> Callback:
        return Callback(step_skip=10)

    def test_record(self, callback: Callback) -> None:
        callback.record(current_step=10, time=0.1)
        np.testing.assert_allclose(callback["time"], [0.1])

    def test_add_meta_info(self, callback: Callback) -> None:
        callback.add_meta_info({"key": "value"})
        assert callback.meta_info["key"] == "value"

    def test_save(self, callback: Callback, tmp_path: Path) -> None:
        callback.record(current_step=10, time=0.1)
        callback.save(filename=tmp_path / "test.hdf5")
        assert (tmp_path / "test.hdf5").exists()
