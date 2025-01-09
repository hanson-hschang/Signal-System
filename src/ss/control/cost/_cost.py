from typing import Any, Callable, Dict, Self, Tuple

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray

from ss.utility.assertion import is_positive_integer, is_positive_number
from ss.utility.assertion.validator import Validator
from ss.utility.callback import Callback
from ss.utility.descriptor import (
    MultiSystemNDArrayDescriptor,
    ReadOnlyDescriptor,
)
from ss.utility.figure import SequenceTrajectoryFigure


class Cost:
    class _TimeStepValidator(Validator):
        def __init__(self, time_step: float) -> None:
            super().__init__()
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
            super().__init__()
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
            super().__init__()
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
        def __init__(self, number_of_systems: int) -> None:
            super().__init__()
            self._number_of_systems = number_of_systems
            self._validate_functions.append(self._validate_value_range)

        def _validate_value_range(self) -> bool:
            if is_positive_integer(self._number_of_systems):
                return True
            self._errors.append(
                f"number_of_systems = {self._number_of_systems} must be a positive integer"
            )
            return False

        def get_value(self) -> int:
            return self._number_of_systems

    def __init__(
        self,
        time_step: float,
        state_dim: int,
        control_dim: int,
        number_of_systems: int = 1,
        **kwargs: Any,
    ) -> None:
        self._time_step = self._TimeStepValidator(time_step).get_value()
        self._state_dim = self._StateDimValidator(state_dim).get_value()
        self._control_dim = self._ControlDimValidator(control_dim).get_value()
        self._number_of_systems = self._NumberOfSystemsValidator(
            number_of_systems
        ).get_value()

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
    state = MultiSystemNDArrayDescriptor("_number_of_systems", "_state_dim")
    control = MultiSystemNDArrayDescriptor("_number_of_systems", "_control_dim")

    def duplicate(self, number_of_systems: int) -> "Cost":
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


class CostCallback(Callback):
    def __init__(
        self,
        step_skip: int,
        cost: Cost,
    ) -> None:
        assert issubclass(type(cost), Cost), "cost must be an instance of Cost."
        self._cost = cost
        super().__init__(step_skip)

    def _record(self, time: float) -> None:
        super()._record(time)
        self._callback_params["cost"].append(self._cost.evaluate().copy())


class CostTrajectoryFigure(SequenceTrajectoryFigure):
    """
    Figure for plotting the cost trajectory.
    """

    def __init__(
        self,
        time_trajectory: ArrayLike,
        cost_trajectory: ArrayLike,
        fig_size: Tuple[int, int] = (12, 8),
    ) -> None:
        cost_trajectory = np.array(cost_trajectory)

        match len(cost_trajectory.shape):
            case 1:
                cost_trajectory = cost_trajectory[np.newaxis, :]
            case _:
                pass
        assert (
            len(cost_trajectory.shape) == 2
        ), "cost_trajectory in general is a 2D array with shape (number_of_systems, time_length)."

        super().__init__(
            time_trajectory,
            number_of_systems=cost_trajectory.shape[0],
            fig_size=fig_size,
            fig_title="Accumulated Cost Trajectory",
        )
        assert cost_trajectory.shape[1] == self._sequence_length, (
            "cost_trajectory must have the same length as time_trajectory."
            "cost_trajectory in general is a 2D array with shape (number_of_systems, time_length)."
        )
        self._cost_trajectory = cost_trajectory
        self._cost_subplot: Axes = self._subplots[0][0]

    def plot(self) -> Self:
        time_step = np.mean(np.diff(self._sequence_trajectory))
        cumsum_cost_trajectory = (
            np.cumsum(self._cost_trajectory, axis=1) * time_step
        )
        if self._number_of_systems <= 10:
            self._plot_each_system_cost_trajectory(
                cumsum_cost_trajectory=cumsum_cost_trajectory,
            )
        else:
            kwargs: Dict = dict(
                color=self._default_color,
                alpha=self._default_alpha,
            )
            self._plot_each_system_cost_trajectory(
                cumsum_cost_trajectory=cumsum_cost_trajectory,
                **kwargs,
            )
            mean_trajectory, std_trajectory = (
                self._compute_system_statistics_trajectory(
                    signal_trajectory=cumsum_cost_trajectory[:, np.newaxis, :],
                )
            )
            self._plot_system_statistics_cost_trajectory(
                mean_trajectory=mean_trajectory,
                std_trajectory=std_trajectory,
            )
        super().plot()
        return self

    def _plot_each_system_cost_trajectory(
        self,
        cumsum_cost_trajectory: NDArray[np.float64],
        **kwargs: Any,
    ) -> None:
        for i in range(self._number_of_systems):
            self._cost_subplot.plot(
                self._sequence_trajectory,
                cumsum_cost_trajectory[i, :],
                **kwargs,
            )
        self._cost_subplot.set_ylabel("Accumulated Cost")
        self._cost_subplot.grid(True)

    def _plot_system_statistics_cost_trajectory(
        self,
        mean_trajectory: NDArray[np.float64],
        std_trajectory: NDArray[np.float64],
    ) -> None:
        self._plot_statistics_signal_trajectory(
            self._cost_subplot,
            mean_trajectory.squeeze(),
            std_trajectory.squeeze(),
        )
