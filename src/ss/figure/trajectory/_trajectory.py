from typing import Any, List, Optional, Self, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from numpy.typing import ArrayLike, NDArray

from ss.figure import Figure
from ss.utility.assertion import is_positive_integer, is_positive_number


class SequenceTrajectoryFigure(Figure):
    def __init__(
        self,
        sequence_trajectory: ArrayLike,
        batch_size: int = 1,
        fig_size: Tuple = (12, 8),
        fig_title: Optional[str] = None,
        fig_layout: Tuple[int, int] = (1, 1),
        column_ratio: Optional[Tuple[float, ...]] = None,
        row_ratio: Optional[Tuple[float, ...]] = None,
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
            batch_size
        ), f"{batch_size = } must be a positive integer."
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

        super().__init__(
            sup_xlabel="sequence",
            fig_size=fig_size,
            fig_title=fig_title,
            fig_layout=fig_layout,
            column_ratio=column_ratio,
            row_ratio=row_ratio,
        )

        self._sequence_trajectory = np.array(sequence_trajectory)
        self._sequence_length = int(sequence_trajectory.shape[0])
        self._batch_size = batch_size
        self._default_color = "C0"
        self._default_alpha = 0.2
        self._default_std_alpha = 0.5
        self._color_map = plt.get_cmap("Greys")

    def _plot_signal_trajectory(
        self,
        ax: Axes,
        signal_trajectory: NDArray[np.float64],
        sequence_trajectory: Optional[NDArray] = None,
        ylabel: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        sequence_trajectory = (
            self._sequence_trajectory
            if sequence_trajectory is None
            else np.array(sequence_trajectory)
        )
        ax.plot(
            sequence_trajectory,
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
        sequence_trajectory: Optional[NDArray] = None,
        ylabel: Optional[str] = None,
    ) -> QuadMesh:
        sequence_trajectory = (
            self._sequence_trajectory
            if sequence_trajectory is None
            else np.array(sequence_trajectory)
        )
        time_horizon = sequence_trajectory[-1] - sequence_trajectory[0]
        time_lim = (
            sequence_trajectory[0] - time_horizon * 0.05,
            sequence_trajectory[-1] + time_horizon * 0.05,
        )
        dimension = probability_trajectory.shape[0]
        for d in range(dimension - 1):
            ax.axhline(d + 0.5, color="black", linewidth=0.5, linestyle="--")

        sequence_grid, probability_grid = np.meshgrid(
            sequence_trajectory,
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
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        return image_mesh


class TimeTrajectoryFigure(SequenceTrajectoryFigure):
    def __init__(
        self,
        time_trajectory: ArrayLike,
        batch_size: int = 1,
        fig_size: Tuple = (12, 8),
        fig_title: Optional[str] = None,
        fig_layout: Tuple[int, int] = (1, 1),
        column_ratio: Optional[Tuple[float, ...]] = None,
        row_ratio: Optional[Tuple[float, ...]] = None,
    ) -> None:
        super().__init__(
            sequence_trajectory=time_trajectory,
            batch_size=batch_size,
            fig_size=fig_size,
            fig_title=fig_title,
            fig_layout=fig_layout,
            column_ratio=column_ratio,
            row_ratio=row_ratio,
        )
        self._time_trajectory = self._sequence_trajectory.copy()
        self._time_horizon = (
            self._time_trajectory[-1] - self._time_trajectory[0]
        )
        self._sup_xlabel = "time (sec)"
