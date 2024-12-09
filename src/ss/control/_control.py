import numpy as np
from numba import njit
from numpy.typing import NDArray

from ss.utility.assertion import is_positive_integer
from ss.utility.descriptor import (
    MultiSystemTensorDescriptor,
    ReadOnlyDescriptor,
)


class Controller:
    def __init__(
        self,
        control_dim: int,
        number_of_systems: int = 1,
    ) -> None:
        assert is_positive_integer(
            control_dim
        ), f"{control_dim = } must be a positive integer"
        assert is_positive_integer(
            number_of_systems
        ), f"{number_of_systems = } must be a positive integer"

        self._control_dim = int(control_dim)
        self._number_of_systems = int(number_of_systems)
        self._control = np.zeros(
            (self._number_of_systems, self._control_dim),
            dtype=np.float64,
        )

    control_dim = ReadOnlyDescriptor[int]()
    number_of_systems = ReadOnlyDescriptor[int]()
    control = MultiSystemTensorDescriptor(
        "_number_of_systems",
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
