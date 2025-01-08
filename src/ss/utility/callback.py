from typing import Any, DefaultDict, Dict, List, Optional, Union

from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ss.utility.assertion.validator import FilePathValidator
from ss.utility.data import MetaData, MetaInfo, MetaInfoValueType


class Callback:
    def __init__(self, step_skip: int) -> None:
        self._data_file_extension = ".hdf5"
        self.sample_every = step_skip
        self._callback_params: DefaultDict[str, List] = defaultdict(list)
        self._meta_data: MetaData = MetaData()
        self._meta_info: MetaInfo = MetaInfo()

    @property
    def meta_data(self) -> MetaData:
        return self._meta_data

    @property
    def meta_info(self) -> MetaInfo:
        return self._meta_info

    def record(self, current_step: int, time: float) -> None:
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

    def add_meta_data(
        self,
        meta_data: Optional[MetaData] = None,
        **meta_data_dict: Union[ArrayLike, MetaData],
    ) -> None:
        """
        Add meta data to the callback.

        Parameters:
        -----------
        meta_data: MetaData
            The meta data to be added to the callback.
        meta_data_dict: Union[ArrayLike, MetaData]
            The meta data to be added to the callback.
        """
        for key, value in meta_data_dict.items():
            self._meta_data[key] = value
        if meta_data is not None:
            for key, value in meta_data.items():
                self._meta_data[key] = value

    def add_meta_info(
        self,
        meta_info: Optional[MetaInfo] = None,
        **meta_info_dict: MetaInfoValueType,
    ) -> None:
        """
        Add meta information to the callback.

        Parameters:
        -----------
        meta_info: MetaInfo
            The meta information to be added to the callback.
        meta_info_dict: MetaInfoValueType
            The meta information to be added to the callback.
        """
        for key, value in meta_info_dict.items():
            self._meta_info[key] = value
        if meta_info is not None:
            for key, value in meta_info.items():
                self._meta_info[key] = value

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
            if len(self._meta_data) > 0:
                self._save_meta_data(
                    f.create_group(MetaData.NAME),
                    self._meta_data,
                )
            for key in self._callback_params.keys():
                f.create_dataset(
                    name=key,
                    data=self[key],
                )
            for key, value in self._meta_info.items():
                f.attrs[key] = value

    @staticmethod
    def _save_meta_data(h5_group: h5py.Group, meta_data: MetaData) -> None:
        for key, value in meta_data.items():
            if isinstance(value, MetaData):
                Callback._save_meta_data(h5_group.create_group(key), value)
            else:
                h5_group.create_dataset(name=key, data=value)
