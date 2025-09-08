from typing import Optional, Self, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray

from ss.figure.matrix import MatrixFigure


class StochasticMatrixFigure(MatrixFigure):
    def __init__(
        self,
        stochastic_matrix: ArrayLike,
        fig_size: Tuple[int, int] = (12, 8),
        fig_title: str = "Stochastic Matrix Analysis",
    ) -> None:
        stochastic_matrix = np.array(stochastic_matrix)
        assert (
            len(stochastic_matrix.shape) == 2
        ), "stochastic_matrix must be a 2D array."
        assert np.all(
            (0 <= stochastic_matrix) & (stochastic_matrix <= 1)
        ) and np.allclose(
            np.sum(stochastic_matrix, axis=1),
            np.ones(stochastic_matrix.shape[0]),
        ), "stochastic_matrix must be a matrix of row probability."

        super().__init__(
            matrix=stochastic_matrix,
            fig_size=fig_size,
            fig_title=fig_title,
        )

    def _plot_eigen_value(self, ax: Axes, matrix: NDArray) -> None:
        super()._plot_eigen_value(ax=ax, matrix=matrix)
        unit_circle = plt.Circle(
            (0, 0), 1, color="gray", linestyle="--", fill=False, alpha=0.5
        )
        ax.add_artist(unit_circle)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

    def plot(self) -> Self:
        return super().plot()
