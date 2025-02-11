from typing import Any, Dict, Self, Union

from pathlib import Path

import h5py
import numpy as np

from ss.utility.assertion.validator import FilePathValidator


class CheckpointInfo(dict):
    _data_file_extension = ".hdf5"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @classmethod
    def load(cls, filename: Union[str, Path]) -> Self:
        filepath = FilePathValidator(
            filename, cls._data_file_extension
        ).get_filepath()
        with h5py.File(filepath, "r") as f:
            checkpoint_info = cls._load(f)
        return cls(**checkpoint_info)

    @classmethod
    def _load(cls, group: h5py.Group) -> Dict[str, Any]:
        checkpoint_info: Dict[str, Any] = dict()
        for key, value in group.items():
            if isinstance(value, h5py.Group):
                checkpoint_info[key] = cls._load(value)
            elif isinstance(value, h5py.Dataset):
                checkpoint_info[key] = np.array(value)
            else:
                checkpoint_info[key] = value
        return checkpoint_info

    def save(self, filename: Union[str, Path]) -> None:
        filepath = FilePathValidator(
            filename, self._data_file_extension
        ).get_filepath()
        with h5py.File(filepath, "w") as f:
            for key, value in self.items():
                self._save(f, key, value)

    @classmethod
    def _save(cls, group: h5py.Group, name: str, value: Any) -> None:
        if isinstance(value, dict):
            subgroup = group.create_group(name)
            for key, val in value.items():
                cls._save(subgroup, key, val)
        elif isinstance(value, (list, tuple, np.ndarray)):
            group.create_dataset(name, data=value)
        else:
            group.attrs[name] = value
