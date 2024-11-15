from typing import Any, DefaultDict, List

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray


class Callback:
    def __init__(
        self,
        step_skip: int,
    ) -> None:
        self.sample_every = step_skip
        self.callback_params: DefaultDict[str, List] = defaultdict(list)

    def make_callback(
        self,
        current_step: int,
        time: float,
    ) -> None:
        if current_step % self.sample_every == 0:
            self._record_params(time)

    def _record_params(self, time: float) -> None:
        self.callback_params["time"].append(time)

    def __getitem__(self, key: str) -> NDArray[np.float64]:
        assert isinstance(key, str), "key must be a string."
        assert (
            key in self.callback_params.keys()
        ), f"{key} not in callback_params."
        signal_trajectory = np.array(self.callback_params[key])
        if len(signal_trajectory.shape) > 1:
            signal_trajectory = np.moveaxis(signal_trajectory, 0, -1)
        return signal_trajectory

    def save_params(self, filename: str) -> None:
        pass
