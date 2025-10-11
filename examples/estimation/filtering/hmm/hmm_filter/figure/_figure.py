from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from ss.figure.trajectory import SequenceTrajectoryFigure


class DualHmmFigure(SequenceTrajectoryFigure):
    def __init__(
        self,
        time_trajectory: ArrayLike,
        estimation_trajectory: ArrayLike,
        dual_estimation_trajectory: ArrayLike,
        fig_size: tuple = (12, 8),
        fig_title: str | None = None,
    ) -> None:
        self._estimation_trajectory = np.array(estimation_trajectory)
        self._dual_estimation_trajectory = np.array(dual_estimation_trajectory)

        self._estimation_dim = self._estimation_trajectory.shape[0]

        fig_layout: tuple = (self._estimation_dim, 1)

        super().__init__(
            sequence_trajectory=time_trajectory,
            fig_size=fig_size,
            fig_layout=fig_layout,
            fig_title=(
                "Dual Hidden Markov Model Filter"
                if fig_title is None
                else fig_title
            ),
        )
        self._sup_xlabel = "time step"

    def plot(self) -> Self:
        for i in range(self._estimation_dim):
            self._subplots[i][0].plot(
                self._sequence_trajectory,
                self._estimation_trajectory[i],
                label="Estimation",
                color="C1",
            )
            self._subplots[i][0].plot(
                self._sequence_trajectory,
                self._dual_estimation_trajectory[i],
                label="Dual Estimation",
                color="C0",
                linestyle="--",
            )
            self._subplots[i][0].set_ylabel(f"State {i}")
            self._subplots[i][0].set_ylim(-0.1, 1.1)

        self._subplots[0][0].legend(
            bbox_to_anchor=(1.0, 1.2), loc="upper right", ncol=2
        )
        super().plot()
        return self
