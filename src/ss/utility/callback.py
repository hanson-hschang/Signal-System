from typing import Any, DefaultDict, Dict, List, Union

from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from ss.utility.assertion.validator import FilePathValidator


class Callback:
    def __init__(
        self,
        step_skip: int,
    ) -> None:
        self._data_file_extension = ".hdf5"
        self.sample_every = step_skip
        self._callback_params: DefaultDict[str, List] = defaultdict(list)
        self._meta_info: DefaultDict[str, Any] = defaultdict()

    @property
    def meta_info(self) -> Dict[str, Any]:
        return self._meta_info

    def record(
        self,
        current_step: int,
        time: float,
    ) -> None:
        if current_step % self.sample_every == 0:
            self._record(time)

    def _record(self, time: float) -> None:
        self._callback_params["time"].append(time)

    def __getitem__(self, key: str) -> NDArray[np.float64]:
        assert isinstance(key, str), "key must be a string."
        assert (
            key in self._callback_params.keys()
        ), f"{key} not in callback parameters."
        signal_trajectory = np.array(self._callback_params[key])
        if len(signal_trajectory.shape) > 1:
            signal_trajectory = np.moveaxis(signal_trajectory, 0, -1)
        return signal_trajectory

    def save(self, filename: Union[str, Path]) -> None:
        """
        Save callback parameters to an HDF5 file.

        Parameters:
        -----------
        filename: str or Path
            The path to the file to save the callback parameters.
        """
        filepath = FilePathValidator(
            filename, self._data_file_extension
        ).get_filepath()

        with h5py.File(filepath, "w") as f:

            for key in self._callback_params.keys():
                data = self[key]

                f.create_dataset(
                    name=key,
                    data=data,
                )
            for key, value in self._meta_info.items():
                f.attrs[key] = value

    def add_meta_info(self, meta_info: Dict[str, Any]) -> None:
        """
        Add meta information to the callback.

        Parameters:
        -----------
        meta_info: Dict[str, Any]
            The meta information to be added to the callback.
        """
        assert isinstance(meta_info, dict), "meta_info must be a dictionary."
        self._meta_info.update(meta_info)
