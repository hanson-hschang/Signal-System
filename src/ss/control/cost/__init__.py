from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from ss.tool.assertion import isPositiveInteger, isPositiveNumber
from ss.tool.descriptor import MultiSystemTensorDescriptor, ReadOnlyDescriptor


class Cost:
    def __init__(
        self,
        time_step: float,
        state_dim: int,
        control_dim: int,
        number_of_systems: int = 1,
        **kwargs: Any,
    ) -> None:
        assert isPositiveNumber(
            time_step
        ), f"time_step {time_step} must be a positive number"
        assert isPositiveInteger(
            state_dim
        ), f"state_dim {state_dim} must be a positive integer"
        assert isPositiveInteger(
            control_dim
        ), f"control_dim {control_dim} must be a positive integer"
        assert isPositiveInteger(
            number_of_systems
        ), f"number_of_systems {number_of_systems} must be a positive integer"

        self._time_step = time_step
        self._state_dim = int(state_dim)
        self._control_dim = int(control_dim)
        self._number_of_systems = int(number_of_systems)

        self._state = np.zeros(
            (self._number_of_systems, self._state_dim), dtype=np.float64
        )
        self._control = np.zeros(
            (self._number_of_systems, self._control_dim), dtype=np.float64
        )
        self._cost = np.zeros(self._number_of_systems, dtype=np.float64)
        self._evaluate: Callable = self._evaluate_running
        super().__init__(**kwargs)

    time_step = ReadOnlyDescriptor[float]()
    state_dim = ReadOnlyDescriptor[int]()
    control_dim = ReadOnlyDescriptor[int]()
    number_of_systems = ReadOnlyDescriptor[int]()
    state = MultiSystemTensorDescriptor("_number_of_systems", "_state_dim")
    control = MultiSystemTensorDescriptor("_number_of_systems", "_control_dim")

    def create_multiple_costs(self, number_of_systems: int) -> "Cost":
        """
        Create multiple costs.

        Parameters
        ----------
        `number_of_systems: int`
            The number of costs to be created.

        Returns
        -------
        `cost: Cost`
            The created multi-cost.
        """
        return self.__class__(
            time_step=self._time_step,
            state_dim=self._state_dim,
            control_dim=self._control_dim,
            number_of_systems=number_of_systems,
        )

    def evaluate(self) -> NDArray[np.float64]:
        self._evaluate()
        cost: NDArray[np.float64] = (
            self._cost[0] if self._number_of_systems == 1 else self._cost
        )
        return cost

    def set_terminal(self, terminal_flag: bool = True) -> None:
        if terminal_flag:
            self._evaluate = self._evaluate_terminal
        else:
            self._evaluate = self._evaluate_running

    def _evaluate_terminal(self) -> None:
        self._cost = np.zeros(self._number_of_systems, dtype=np.float64)

    def _evaluate_running(self) -> None:
        self._cost = np.zeros(self._number_of_systems, dtype=np.float64)
