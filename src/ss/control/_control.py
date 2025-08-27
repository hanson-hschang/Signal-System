import numpy as np
from numba import njit
from numpy.typing import NDArray

from ss.utility.assertion import is_positive_integer
from ss.utility.descriptor import (
    BatchNDArrayDescriptor,
    ReadOnlyDescriptor,
)


class Controller:
    def __init__(
        self,
        control_dim: int,
        batch_size: int = 1,
    ) -> None:
        assert is_positive_integer(
            control_dim
        ), f"{control_dim = } must be a positive integer"
        assert is_positive_integer(
            batch_size
        ), f"{batch_size = } must be a positive integer"

        self._control_dim = int(control_dim)
        self._batch_size = int(batch_size)
        self._control = np.zeros(
            (self._batch_size, self._control_dim),
            dtype=np.float64,
        )

    control_dim = ReadOnlyDescriptor[int]()
    batch_size = ReadOnlyDescriptor[int]()
    control = BatchNDArrayDescriptor(
        "_batch_size",
        "_control_dim",
    )

    def compute_control(self) -> None:
        self._update(
            self._control,
            self._compute_control(),
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _update(
        array: NDArray[np.float64],
        process: NDArray[np.float64],
    ) -> None:
        array[...] = process

    def _compute_control(self) -> NDArray[np.float64]:
        return np.zeros_like(self._control)
