from typing import Any, DefaultDict, List, Optional, Self, Union, cast

from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ss.utility.assertion.validator import FilePathValidator
from ss.utility.data import MetaData, MetaInfo, MetaInfoValueType


class Callback:
    FILE_EXTENSION = ".hdf5"

    def __init__(self, step_skip: int) -> None:
        self._data_file_extension = ".hdf5"
        self.sample_every = int(step_skip)
        self._callback_params: DefaultDict[str, List] = defaultdict(list)
        self._meta_data: MetaData = MetaData()
        self._meta_info: MetaInfo = MetaInfo()
        self._meta_info["__step_skip__"] = int(self.sample_every)

    @property
    def meta_data(self) -> MetaData:
        return self._meta_data

    @property
    def meta_info(self) -> MetaInfo:
        return self._meta_info

    def record(self, current_step: int, time: float) -> None:
        if current_step % self.sample_every == 0:
            self._record(time)
        # TODO: add key name violation check

    def _record(self, time: float) -> None:
        self._callback_params["time"].append(time)

    def __getitem__(self, key: str) -> NDArray[np.float64]:
        assert isinstance(key, str), "key must be a string."
        assert (
            key in self._callback_params.keys()
        ), f"{key} not in callback parameters."
        signal_trajectory = self.to_numpy_array(self._callback_params[key])
        # signal_trajectory = np.array(self._callback_params[key])
        # if len(signal_trajectory.shape) > 1:
        #     signal_trajectory = np.moveaxis(signal_trajectory, 0, -1)
        return signal_trajectory

    @staticmethod
    def to_numpy_array(value: List) -> NDArray:
        signal_trajectory = np.array(value)
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

        Arguments:
        -----------
            meta_data: Optional[MetaData] = None
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

        Arguments:
        ----------
            meta_info: Optional[MetaInfo] = None
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

        Arguments:
        ----------
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
            # for key in self._callback_params.keys():
            #     f.create_dataset(
            #         name=key,
            #         data=self[key],
            #     )
            for key, value in self._callback_params.items():
                self._save_callback_params(f, key, value)
            for key, value in self._meta_info.items():
                self._save_meta_info(f, key, cast(MetaInfoValueType, value))
                # f.attrs[key] = value

    @classmethod
    def _save_meta_info(
        cls, h5_group: h5py.Group, key: str, value: MetaInfoValueType
    ) -> None:
        if "/" in key:
            parent_key, child_key = key.split("/")
            h5_subgroup = (
                h5_group.create_group(parent_key)
                if parent_key not in h5_group
                else h5_group[parent_key]
            )
            if isinstance(h5_subgroup, h5py.Group):
                cls._save_meta_info(h5_subgroup, child_key, value)
            elif isinstance(h5_subgroup, h5py.Dataset):
                if "/" in child_key:
                    # This is a nested meta info
                    # TODO: Implement this
                    pass
                else:
                    h5_subgroup.attrs[child_key] = value
        else:
            h5_group.attrs[key] = value

    @classmethod
    def _save_meta_data(
        cls, h5_group: h5py.Group, meta_data: MetaData
    ) -> None:
        for key, value in meta_data.items():
            if isinstance(value, MetaData):
                cls._save_meta_data(h5_group.create_group(key), value)
            else:
                h5_group.create_dataset(name=key, data=value)

    @classmethod
    def _save_callback_params(
        cls,
        h5_group: h5py.Group,
        key: str,
        value: List,
    ) -> None:
        if "/" in key:
            parent_key, child_key = key.split("/")

            h5_subgroup = (
                h5_group.create_group(parent_key)
                if parent_key not in h5_group
                else h5_group[parent_key]
            )
            if isinstance(h5_subgroup, h5py.Group):
                Callback._save_callback_params(h5_subgroup, child_key, value)
            elif isinstance(h5_subgroup, h5py.Dataset):
                # If the parent key already exists as a dataset, convert it to a dataset
                # under the parent key group with the same name as the parent key.
                temp_value = np.array(h5_subgroup)
                del h5_group[parent_key]
                h5_subgroup = h5_group.create_group(parent_key)
                h5_subgroup.create_dataset(parent_key, data=temp_value)

        else:
            h5_group.create_dataset(
                name=key,
                data=cls.to_numpy_array(value),
            )

    @classmethod
    def load(cls, filename: Union[str, Path]) -> Self:
        filepath = FilePathValidator(
            filename, cls.FILE_EXTENSION
        ).get_filepath()
        with h5py.File(filepath, "r") as f:
            callback = cls(cast(int, f.attrs["__step_skip__"]))
            callback._load_meta_info(f)
            if MetaData.NAME in f:
                callback._load_meta_data(cast(h5py.Group, f[MetaData.NAME]))
            callback._load_callback_params(f)
        return callback

    def _load_meta_info(self, h5_file: h5py.File) -> None:
        for key, value in h5_file.attrs.items():
            if key == "__step_skip__":
                continue
            match type(value):
                case np.int64:
                    self._meta_info[key] = int(value)
                case np.float64:
                    self._meta_info[key] = float(value)

    def _load_meta_data(self, h5_group: h5py.Group) -> None:
        for key, value in h5_group.items():
            if isinstance(value, h5py.Group):
                self._meta_data[key] = MetaData.from_hdf5_group(value)
            elif isinstance(value, h5py.Dataset):
                self._meta_data[key] = np.array(value)

    def _load_callback_params(self, h5_group: h5py.Group) -> None:
        for key, value in h5_group.items():
            if key == MetaData.NAME:
                continue
            signal_trajectory = np.array(value)
            for k in range(signal_trajectory.shape[-1]):
                self._callback_params[key].append(signal_trajectory[..., k])
