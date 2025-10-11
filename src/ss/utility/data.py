from collections.abc import KeysView
from pathlib import Path
from typing import Union, get_args

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ss.utility.assertion.validator import (
    FilePathExistenceValidator,
    SignalTrajectoryValidator,
    Validator,
)
from ss.utility.type import from_numpy_generic, get_type_string


class MetaData(dict):
    NAME = "__meta_data__"

    class _MetaDataValidator(Validator):
        def __init__(
            self, meta_data: dict[str, Union[ArrayLike, "MetaData"]]
        ) -> None:
            super().__init__(meta_data)
            self._meta_data = meta_data
            self._validated_meta_data: dict[
                str, Union[NDArray, "MetaData"]
            ] = dict()
            self.add_validation(self._validate_meta_data)

        def _validate_meta_data(self) -> bool:
            if not isinstance(self._meta_data, dict):
                self.add_error(
                    "meta_data must be a dictionary.",
                    f"{type(self._meta_data) = }",
                )
                return False
            for key, value in self._meta_data.items():
                if not isinstance(key, str):
                    self.add_error(
                        "each key in meta_data must be a string.",
                        f"{self._meta_data.keys() = }",
                    )
                    return False
                if isinstance(value, MetaData):
                    self._validated_meta_data[key] = value
                elif isinstance(value, (list, tuple, np.ndarray)):
                    self._validated_meta_data[key] = np.array(value)
                else:
                    self.add_error(
                        "value in meta_data must be "
                        "an array-like or a MetaData.",
                        f"meta_data[{key}] = {value!r}",
                    )
                    return False
            return True

        def get_meta_data(self) -> dict[str, Union[NDArray, "MetaData"]]:
            return self._validated_meta_data

    def __init__(self, **meta_data: Union[ArrayLike, "MetaData"]) -> None:
        _meta_data = self._MetaDataValidator(meta_data).get_meta_data()
        super().__init__(**_meta_data)

    def __setitem__(
        self, key: str, value: Union[ArrayLike, "MetaData"]
    ) -> None:
        super().__setitem__(
            key, self._MetaDataValidator({key: value}).get_meta_data()[key]
        )

    @classmethod
    def from_hdf5_group(cls, h5_group: h5py.Group) -> "MetaData":
        meta_data = MetaData()
        for key, value in h5_group.items():
            if isinstance(value, h5py.Group):
                meta_data[key] = cls.from_hdf5_group(value)
            else:
                meta_data[key] = np.array(value)
        return cls(**meta_data)


MetaInfoValueType = Union[bool, int, float, str]


class MetaInfo(dict):
    class _MetaInfoValidator(Validator):
        def __init__(self, meta_info: dict[str, MetaInfoValueType]) -> None:
            super().__init__(meta_info)
            self._meta_info = meta_info
            self.add_validation(self._validate_meta_info)

        def _validate_meta_info(self) -> bool:
            if not isinstance(self._meta_info, dict):
                self.add_error(
                    "meta_info must be a dictionary.",
                    f"{type(self._meta_info) = }",
                )
                return False
            allowed_types = get_args(MetaInfoValueType)
            allowed_types_str = get_type_string(allowed_types)
            for key, value in self._meta_info.items():
                if not isinstance(key, str):
                    self.add_error(
                        "each key in meta_info must be a string.",
                        f"{self._meta_info.keys() = }",
                    )
                    return False
                if not isinstance(value, allowed_types):
                    self.add_error(
                        f"each value in meta_info must "
                        f"be a {allowed_types_str}",
                        f"meta_info[{key}] = {value} with "
                        f"type({value}) = {type(value)}",
                    )
                    return False
            return True

        def get_meta_info(self) -> dict[str, MetaInfoValueType]:
            return self._meta_info

    def __init__(self, **meta_info: MetaInfoValueType) -> None:
        _meta_info = self._MetaInfoValidator(meta_info).get_meta_info()
        super().__init__(**_meta_info)

    def __setitem__(self, key: str, value: MetaInfoValueType) -> None:
        meta_info = {key: value}
        super().__setitem__(
            key, self._MetaInfoValidator(meta_info).get_meta_info()[key]
        )

    @classmethod
    def from_hdf5_attrs(cls, h5_attrs: h5py.AttributeManager) -> "MetaInfo":
        meta_info = dict()
        for key, value in h5_attrs.items():
            meta_info[key] = from_numpy_generic(value)
        return cls(**meta_info)


class Data:
    _file_extension = ".hdf5"

    def __init__(
        self,
        signal_trajectory: dict[str, ArrayLike],
        meta_data: MetaData | None = None,
        meta_info: MetaInfo | None = None,
    ) -> None:
        self._signal_trajectory = SignalTrajectoryValidator(
            signal_trajectory
        ).get_trajectory()
        self.meta_data = MetaData() if meta_data is None else meta_data
        self.meta_info = MetaInfo() if meta_info is None else meta_info

    @classmethod
    def load(cls, filename: str | Path) -> "Data":
        filepath = FilePathExistenceValidator(
            filename, cls._file_extension
        ).get_filepath()

        signal_trajectory: dict[str, ArrayLike] = dict()
        meta_data = None
        meta_info = None
        with h5py.File(filepath, "r") as f:
            for key, value in f.items():
                if key == MetaData.NAME and isinstance(
                    meta_data_h5 := f[key], h5py.Group
                ):
                    meta_data = MetaData.from_hdf5_group(meta_data_h5)
                else:
                    signal_trajectory[key] = np.array(value)

            meta_info = MetaInfo.from_hdf5_attrs(f.attrs)

        return cls(signal_trajectory, meta_data, meta_info)

    def __getitem__(self, key: str) -> NDArray[np.float64]:
        return self._signal_trajectory[key]

    def __setitem__(self, key: str, value: ArrayLike) -> None:
        value = np.array(value, dtype=np.float64)
        time_horizon = self._signal_trajectory["time"].shape[0]
        assert value.shape[-1] == time_horizon, (
            "last dimension of value must have the same as 'time_horizon'."
            f"{value.shape[-1] = } does not match the {time_horizon = }"
        )
        self._signal_trajectory[key] = value

    def __delitem__(self, key: str) -> None:
        del self._signal_trajectory[key]

    def __contains__(self, key: str) -> bool:
        return key in self._signal_trajectory

    def keys(self) -> KeysView[str]:
        return self._signal_trajectory.keys()
