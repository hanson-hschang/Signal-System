from typing import Any, List, Optional, Self, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from numpy.typing import ArrayLike, NDArray

from ss.utility.assertion import is_positive_integer, is_positive_number


class SequenceTrajectoryFigure:
    def __init__(
        self,
        sequence_trajectory: ArrayLike,
        number_of_systems: int = 1,
        fig_size: Tuple = (12, 8),
        fig_title: Optional[str] = None,
        fig_layout: Tuple[int, int] = (1, 1),
    ) -> None:
        sequence_trajectory = np.array(sequence_trajectory)
        assert sequence_trajectory.ndim == 1, (
            f"sequency_trajectory must be in the shape of (sequence_length,). "
            f"sequency_trajectory given has the shape of {sequence_trajectory.shape}."
        )
        assert np.all(np.diff(sequence_trajectory) > 0), (
            f"sequency_trajectory must be monotonically increasing. "
            f"sequency_trajectory given is {sequence_trajectory}."
        )
        assert is_positive_integer(
            number_of_systems
        ), f"{number_of_systems = } must be a positive integer."
        assert (
            len(fig_size) == 2
        ), f"{fig_size = } must be a tuple (width, height)."
        assert np.all(
            [is_positive_number(fig_size[0]), is_positive_number(fig_size[1])]
        ), f"values of {fig_size = } must be positive numbers."
        assert (
            len(fig_layout) == 2
        ), f"{fig_layout = } must be a tuple (nrows, ncols)."
        assert np.all(
            [
                is_positive_integer(fig_layout[0]),
                is_positive_integer(fig_layout[1]),
            ]
        ), f"values of {fig_layout = } must be positive integers."

        self._sequence_trajectory = sequence_trajectory
        self._sequence_length = sequence_trajectory.shape[0]
        self._number_of_systems = number_of_systems
        self._default_color = "C0"
        self._default_alpha = 0.2
        self._default_std_alpha = 0.5

        self._fig_size = fig_size
        self._fig_title = fig_title
        self._fig_layout = fig_layout

        self._fig = plt.figure(figsize=self._fig_size)
        self._grid_spec = gridspec.GridSpec(*self._fig_layout, figure=self._fig)
        self._subplots: List[List[Axes]] = []
        for row in range(self._fig_layout[0]):
            self._subplots.append([])
            for col in range(self._fig_layout[1]):
                self._subplots[row].append(
                    self._fig.add_subplot(self._grid_spec[row, col])
                )

        self._sup_xlabel = "sequence"
        self._color_map = plt.get_cmap("Greys")

    def plot(self) -> Self:
        if self._fig_title is not None:
            self._fig.suptitle(self._fig_title)
        self._fig.supxlabel(self._sup_xlabel)
        self._fig.tight_layout()
        return self

    def _plot_signal_trajectory(
        self,
        ax: Axes,
        signal_trajectory: NDArray[np.float64],
        ylabel: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        ax.plot(
            self._sequence_trajectory,
            signal_trajectory,
            **kwargs,
        )
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        ax.grid(True)

    def _compute_system_statistics_trajectory(
        self,
        signal_trajectory: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        mean_trajectory = np.mean(signal_trajectory, axis=0)
        std_trajectory = np.std(signal_trajectory, axis=0)
        return mean_trajectory, std_trajectory

    def _plot_statistics_signal_trajectory(
        self,
        ax: Axes,
        mean_trajectory: NDArray[np.float64],
        std_trajectory: NDArray[np.float64],
    ) -> None:
        ax.plot(
            self._sequence_trajectory,
            mean_trajectory,
        )
        ax.fill_between(
            self._sequence_trajectory,
            mean_trajectory - std_trajectory,
            mean_trajectory + std_trajectory,
            alpha=self._default_std_alpha,
        )

    def _plot_probability_flow(
        self,
        ax: Axes,
        probability_trajectory: NDArray[np.float64],
    ) -> QuadMesh:
        time_horizon = (
            self._sequence_trajectory[-1] - self._sequence_trajectory[0]
        )
        time_lim = (
            self._sequence_trajectory[0] - time_horizon * 0.05,
            self._sequence_trajectory[-1] + time_horizon * 0.05,
        )
        dimension = probability_trajectory.shape[0]
        for d in range(dimension - 1):
            ax.axhline(d + 0.5, color="black", linewidth=0.5, linestyle="--")

        sequence_grid, probability_grid = np.meshgrid(
            self._sequence_trajectory,
            np.arange(dimension),
        )
        image_mesh = ax.pcolormesh(
            sequence_grid,
            probability_grid,
            probability_trajectory,
            cmap=self._color_map,
            vmin=0,
            vmax=1,
        )
        ax.invert_yaxis()
        ax.set_xlim(time_lim)
        ax.set_yticks(np.arange(dimension))
        return image_mesh


class TimeTrajectoryFigure(SequenceTrajectoryFigure):
    def __init__(
        self,
        time_trajectory: ArrayLike,
        number_of_systems: int = 1,
        fig_size: Tuple = (12, 8),
        fig_title: Optional[str] = None,
        fig_layout: Tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__(
            sequence_trajectory=time_trajectory,
            number_of_systems=number_of_systems,
            fig_size=fig_size,
            fig_title=fig_title,
            fig_layout=fig_layout,
        )
        self._sup_xlabel = "Time (sec)"
