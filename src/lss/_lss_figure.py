from typing import Optional, Self, Tuple

import numpy as np
from numpy.typing import ArrayLike

from ss.utility.figure import SequenceTrajectoryFigure


class IterationFigure(SequenceTrajectoryFigure):
    def __init__(
        self,
        iteration_trajectory: ArrayLike,
        training_cost_trajectory: ArrayLike,
        fig_size: Tuple = (12, 8),
        fig_title: Optional[str] = None,
        fig_layout: Tuple[int, int] = (1, 1),
    ) -> None:
        training_cost_trajectory = np.array(training_cost_trajectory)
        match len(training_cost_trajectory.shape):
            case 1:
                training_cost_trajectory = training_cost_trajectory[
                    np.newaxis, :
                ]
            case _:
                pass
        assert (
            len(training_cost_trajectory.shape) == 2
        ), "training_cost_trajectory must be a 2D array with shape (number_of_trainings, sequence_length)."
        super().__init__(
            sequence_trajectory=iteration_trajectory,
            number_of_systems=training_cost_trajectory.shape[0],
            fig_size=fig_size,
            fig_title=fig_title,
            fig_layout=fig_layout,
        )
        self._sup_xlabel = "iteration"
        self._loss_plot = self._subplots[0][0]

        self._training_cost_trajectory = training_cost_trajectory

    def plot(self) -> Self:
        if self._number_of_systems <= 10:
            self._plot_each_training_trajectory()
        else:
            self._plot_average_trajectory()
        super().plot()
        return self

    def _plot_each_training_trajectory(self) -> None:
        for i in range(self._number_of_systems):
            self._plot_signal_trajectory(
                self._loss_plot, self._training_cost_trajectory[i], label="loss"
            )

    def _plot_average_trajectory(self) -> None:
        pass
