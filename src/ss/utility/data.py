from typing import Any, Dict, Optional, Union

from pathlib import Path

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ss.utility.assertion.validator import (
    FilePathExistenceValidator,
    SignalTrajectoryValidator,
)


class MetaInfo:
    def __init__(self, meta_info: Optional[Dict[str, Any]] = None) -> None:
        if meta_info is None:
            meta_info = dict()
        self._meta_info = meta_info

    def __getitem__(self, key: str) -> Any:
        return self._meta_info[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._meta_info[key] = value

    def __delitem__(self, key: str) -> None:
        del self._meta_info[key]

    def __contains__(self, key: str) -> bool:
        return key in self._meta_info

    def __len__(self) -> int:
        return len(self._meta_info)

    def __str__(self) -> str:
        return str(self._meta_info)

    def __repr__(self) -> str:
        return repr(self._meta_info)


class Data:
    _file_extension = ".hdf5"

    def __init__(
        self,
        signal_trajectory: Dict[str, ArrayLike],
        meta_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._signal_trajectory = SignalTrajectoryValidator(
            signal_trajectory
        ).get_trajectory()
        self.meta_info = MetaInfo(meta_info)

    @classmethod
    def load(cls, filename: Union[str, Path]) -> "Data":
        filepath = FilePathExistenceValidator(
            filename, cls._file_extension
        ).get_filepath()

        signal_trajectory: Dict[str, ArrayLike] = dict()
        meta_info = dict()
        with h5py.File(filepath, "r") as f:

            for key, value in f.items():
                signal_trajectory[key] = np.array(value)

            for key, value in f.attrs.items():
                meta_info[key] = value

        return cls(signal_trajectory, meta_info)

    def __getitem__(self, key: str) -> NDArray[np.float64]:
        return self._signal_trajectory[key]

    def __setitem__(self, key: str, value: ArrayLike) -> None:
        value = np.array(value, dtype=np.float64)
        time_horizon = self._signal_trajectory["time"].shape[0]
        assert value.shape[-1] == time_horizon, (
            f"last dimension of value must have the same time_horizon as 'time'."
            f"{value.shape[-1] = } does not match the time_horizon {time_horizon}"
        )
        self._signal_trajectory[key] = value

    def __delitem__(self, key: str) -> None:
        del self._signal_trajectory[key]

    def __contains__(self, key: str) -> bool:
        return key in self._signal_trajectory

    def __len__(self) -> int:
        return len(self._signal_trajectory)
