from typing import Any, Callable, Dict, Self, Tuple

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray

from ss.utility.assertion import is_positive_integer, is_positive_number
from ss.utility.assertion.validator import Validator
from ss.utility.callback import Callback
from ss.utility.descriptor import (
    BatchNDArrayDescriptor,
    ReadOnlyDescriptor,
)


class Cost:
    class _TimeStepValidator(Validator):
        def __init__(self, time_step: float) -> None:
            super().__init__(time_step)
            self._time_step = time_step
            self._validate_functions.append(self._validate_value_range)

        def _validate_value_range(self) -> bool:
            if is_positive_number(self._time_step):
                return True
            self._errors.append(
                f"time_step = {self._time_step} must be a positive number"
            )
            return False

        def get_value(self) -> float:
            return self._time_step

    class _StateDimValidator(Validator):
        def __init__(self, state_dim: int) -> None:
            super().__init__(state_dim)
            self._state_dim = state_dim
            self._validate_functions.append(self._validate_value_range)

        def _validate_value_range(self) -> bool:
            if is_positive_integer(self._state_dim):
                return True
            self._errors.append(
                f"state_dim = {self._state_dim} must be a positive integer"
            )
            return False

        def get_value(self) -> int:
            return self._state_dim

    class _ControlDimValidator(Validator):
        def __init__(self, control_dim: int) -> None:
            super().__init__(control_dim)
            self._control_dim = control_dim
            self._validate_functions.append(self._validate_value_range)

        def _validate_value_range(self) -> bool:
            if is_positive_integer(self._control_dim):
                return True
            self._errors.append(
                f"control_dim = {self._control_dim} must be a positive integer"
            )
            return False

        def get_value(self) -> int:
            return self._control_dim

    class _NumberOfSystemsValidator(Validator):
        def __init__(self, batch_size: int) -> None:
            super().__init__(batch_size)
            self._batch_size = batch_size
            self._validate_functions.append(self._validate_value_range)

        def _validate_value_range(self) -> bool:
            if is_positive_integer(self._batch_size):
                return True
            self._errors.append(
                f"batch_size = {self._batch_size} must be a positive integer"
            )
            return False

        def get_value(self) -> int:
            return self._batch_size

    def __init__(
        self,
        time_step: float,
        state_dim: int,
        control_dim: int,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        self._time_step = self._TimeStepValidator(time_step).get_value()
        self._state_dim = self._StateDimValidator(state_dim).get_value()
        self._control_dim = self._ControlDimValidator(control_dim).get_value()
        self._batch_size = self._NumberOfSystemsValidator(
            batch_size
        ).get_value()

        self._state = np.zeros(
            (self._batch_size, self._state_dim), dtype=np.float64
        )
        self._control = np.zeros(
            (self._batch_size, self._control_dim), dtype=np.float64
        )
        self._cost = np.zeros(self._batch_size, dtype=np.float64)
        self._evaluate: Callable = self._evaluate_running
        super().__init__(**kwargs)

    time_step = ReadOnlyDescriptor[float]()
    state_dim = ReadOnlyDescriptor[int]()
    control_dim = ReadOnlyDescriptor[int]()
    batch_size = ReadOnlyDescriptor[int]()
    state = BatchNDArrayDescriptor("_batch_size", "_state_dim")
    control = BatchNDArrayDescriptor("_batch_size", "_control_dim")

    def duplicate(self, batch_size: int) -> "Cost":
        """
        Create multiple costs.

        Parameters
        ----------
        `batch_size: int`
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
            batch_size=batch_size,
        )

    def evaluate(self) -> NDArray[np.float64]:
        self._evaluate()
        cost: NDArray[np.float64] = (
            self._cost[0] if self._batch_size == 1 else self._cost
        )
        return cost

    def set_terminal(self, terminal_flag: bool = True) -> None:
        if terminal_flag:
            self._evaluate = self._evaluate_terminal
        else:
            self._evaluate = self._evaluate_running

    def _evaluate_terminal(self) -> None:
        self._cost = np.zeros(self._batch_size, dtype=np.float64)

    def _evaluate_running(self) -> None:
        self._cost = np.zeros(self._batch_size, dtype=np.float64)


class CostCallback(Callback):
    def __init__(
        self,
        step_skip: int,
        cost: Cost,
    ) -> None:
        assert issubclass(
            type(cost), Cost
        ), "cost must be an instance of Cost."
        self._cost = cost
        super().__init__(step_skip)

    def _record(self, time: float) -> None:
        super()._record(time)
        self._callback_params["cost"].append(self._cost.evaluate().copy())
