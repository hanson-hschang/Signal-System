from typing import Optional, Self, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from numpy.typing import ArrayLike, NDArray

from ss.figure import Figure


class MatrixFigure(Figure):
    def __init__(
        self,
        matrix: ArrayLike,
        fig_size: Tuple[int, int] = (12, 8),
        fig_title: str = "Matrix Analysis",
    ) -> None:
        matrix = np.array(matrix)
        assert len(matrix.shape) == 2, "matrix must be a 2D array."
        assert np.all((0 <= matrix) & (matrix <= 1)) and np.allclose(
            np.sum(matrix, axis=1), np.ones(matrix.shape[0])
        ), "stochastic_matrix must be a matrix of row probability."

        super().__init__(
            fig_size=fig_size,
            fig_title=fig_title,
            fig_layout=(2, 2),
        )

        self._matrix = matrix
        self._number_of_rows = self._matrix.shape[0]

        self._matrix_subplot = self._subplots[0][0]
        self._eigen_value_subplot = self._subplots[0][1]
        self._singular_value_subplot = self._subplots[1][1]

    def _plot_matrix(self, ax: Axes, matrix: NDArray) -> None:
        assert len(matrix.shape) == 2, "matrix must be a 2D array."
        cax = ax.matshow(
            matrix, cmap="Purples", norm=Normalize(vmin=0, vmax=1)
        )
        plt.colorbar(cax, ax=ax)

    def _plot_eigen_value(self, ax: Axes, matrix: NDArray) -> None:
        try:
            assert (
                len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]
            ), "matrix must be a 2D square matrix."
            eigen_values = np.linalg.eigvals(matrix)
            ax.scatter(eigen_values.real, eigen_values.imag)
        except AssertionError:
            pass
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_title("Eigen Values")
        ax.set_aspect("equal", "box")

    def _plot_singular_value(self, ax: Axes, matrix: NDArray) -> None:
        assert len(matrix.shape) == 2, "matrix must be a 2D matrix."
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        max_singular_value = float(np.max(singular_values))
        ax.axhline(y=0, linestyle="--", color="gray", alpha=0.5)
        ax.scatter(range(1, len(singular_values) + 1), singular_values)
        ax.set_xlabel("Index")
        ax.set_ylabel("Singular Value")
        ax.set_ylim(-max_singular_value * 0.1, max_singular_value * 1.1)

    def plot(self) -> Self:
        self._plot_matrix(ax=self._matrix_subplot, matrix=self._matrix)
        self._plot_eigen_value(
            ax=self._eigen_value_subplot, matrix=self._matrix
        )
        self._plot_singular_value(
            ax=self._singular_value_subplot, matrix=self._matrix
        )
        return super().plot()
