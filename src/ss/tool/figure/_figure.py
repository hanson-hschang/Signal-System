from typing import List, Optional, Self, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ss.tool.assertion import isPositiveInteger, isPositiveNumber


class TimeTrajectoryFigure:
    def __init__(
        self,
        time_trajectory: NDArray[np.float64],
        fig_size: Tuple[int, int] = (12, 8),
        fig_title: Optional[str] = None,
        fig_layout: Tuple[int, int] = (1, 1),
    ) -> None:
        assert (
            len(time_trajectory.shape) == 1
        ), "time_trajectory must be a 1D array."
        assert np.all(
            np.diff(time_trajectory) > 0
        ), "time_trajectory must be monotonically increasing."
        assert len(fig_size) == 2, "fig_size must be a tuple of two values."
        assert isPositiveNumber(fig_size[0]) and isPositiveNumber(
            fig_size[1]
        ), "values of fig_size must be positive numbers."
        assert len(fig_layout) == 2, "fig_layout must be a tuple of two values."
        assert isPositiveInteger(fig_layout[0]) and isPositiveInteger(
            fig_layout[1]
        ), "values of fig_layout must be positive integers."

        self._time_trajectory = time_trajectory
        self._time_length = time_trajectory.shape[0]
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

    def plot_figure(
        self,
    ) -> Self:
        if self._fig_title is not None:
            self._fig.suptitle(self._fig_title)
        self._fig.tight_layout()
        return self

    def show_figure(
        self,
    ) -> None:
        plt.show()
