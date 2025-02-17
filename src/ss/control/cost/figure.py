from typing import Any, Callable, Dict, Self, Tuple

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray

from ss import figure as Figure
from ss.figure import show
from ss.utility.assertion import is_positive_integer, is_positive_number
from ss.utility.assertion.validator import Validator
from ss.utility.callback import Callback
from ss.utility.descriptor import (
    MultiSystemNDArrayDescriptor,
    ReadOnlyDescriptor,
)


class CostTrajectoryFigure(Figure.SequenceTrajectoryFigure):
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
            (
                mean_trajectory,
                std_trajectory,
            ) = self._compute_system_statistics_trajectory(
                signal_trajectory=cumsum_cost_trajectory[:, np.newaxis, :],
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
