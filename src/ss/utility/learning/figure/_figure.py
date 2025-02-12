from typing import Dict, Optional, Self, Tuple

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ss import figure as Figure


class IterationFigure(Figure.SequenceTrajectoryFigure):
    def __init__(
        self,
        training_loss_trajectory: Dict[str, NDArray],
        validation_loss_trajectory: Optional[Dict[str, NDArray]] = None,
        scaling: float = 1.0,
        fig_size: Tuple = (12, 8),
        fig_title: Optional[str] = None,
        fig_layout: Tuple[int, int] = (1, 1),
    ) -> None:
        match training_loss_trajectory["loss"].ndim:
            case 1:
                training_loss_trajectory["loss"] = training_loss_trajectory[
                    "loss"
                ][np.newaxis, :]
            case _:
                pass
        assert (
            len(training_loss_trajectory["loss"].shape) == 2
        ), "training_loss_trajectory['loss'] must be a 2D array with shape (number_of_trainings, iteration_length)."
        super().__init__(
            sequence_trajectory=training_loss_trajectory["iteration"],
            number_of_systems=training_loss_trajectory["loss"].shape[0],
            fig_size=fig_size,
            fig_title=fig_title,
            fig_layout=fig_layout,
        )
        self._sup_xlabel = "iteration"
        self._loss_plot = self._subplots[0][0]

        self._training_loss_trajectory = training_loss_trajectory
        if validation_loss_trajectory is not None:
            match validation_loss_trajectory["loss"].ndim:
                case 1:
                    validation_loss_trajectory["loss"] = (
                        validation_loss_trajectory["loss"][np.newaxis, :]
                    )
                case _:
                    pass
            assert (
                len(validation_loss_trajectory["loss"].shape) == 2
            ), "validation_loss_trajectory['loss'] must be a 2D array with shape (number_of_trainings, iteration_length)"
        self._validation_loss_trajectory = validation_loss_trajectory
        self._scaling = scaling

    @property
    def loss_plot_ax(self) -> Axes:
        return self._loss_plot

    def plot(self) -> Self:
        if self._number_of_systems == 1:
            self._plot_training_trajectory()
        else:
            self._plot_statistic_training_trajectory()
        super().plot()
        return self

    def _plot_training_trajectory(self) -> None:
        self._plot_signal_trajectory(
            self._loss_plot,
            self._training_loss_trajectory["loss"][0] * self._scaling,
            ylabel="loss",
            label="training",
        )
        if self._validation_loss_trajectory is not None:
            iteration_trajectory = self._validation_loss_trajectory[
                "iteration"
            ]
            validation_loss_trajectory = self._validation_loss_trajectory[
                "loss"
            ]
            self._loss_plot.scatter(
                iteration_trajectory,
                validation_loss_trajectory[0, :] * self._scaling,
                label="validation",
                color="C1",
                s=100,
                zorder=2,
            )
        self._loss_plot.legend()
        # self._loss_plot.set_xscale('log')

    def _plot_statistic_training_trajectory(self) -> None:
        pass
