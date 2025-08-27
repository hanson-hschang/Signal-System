from typing import Dict, List, Optional, Self, Tuple

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ss.figure.trajectory import SequenceTrajectoryFigure


class IterationFigure(SequenceTrajectoryFigure):
    def __init__(
        self,
        training_loss_history: Dict[str, List],
        validation_loss_history: Optional[Dict[str, List]] = None,
        iteration_log_scale: bool = True,
        scaling: float = 1.0,
        fig_size: Tuple = (12, 8),
        fig_title: Optional[str] = None,
        fig_layout: Tuple[int, int] = (1, 1),
    ) -> None:
        training_loss = np.array(training_loss_history["loss"])
        match training_loss.ndim:
            case 1:
                training_loss = training_loss[np.newaxis, :]
            case _:
                pass
        assert len(training_loss.shape) == 2, (
            "training_loss_history['loss'] must be a 2D array "
            "with shape (number_of_trainings, iteration_length)."
        )
        _training_loss_history: Dict[str, NDArray] = dict(
            iteration=np.array(training_loss_history["iteration"]),
            loss=training_loss,
        )
        super().__init__(
            sequence_trajectory=_training_loss_history["iteration"],
            batch_size=training_loss.shape[0],
            fig_size=fig_size,
            fig_title=fig_title,
            fig_layout=fig_layout,
        )
        self._iteration_log_scale = iteration_log_scale
        self._sup_xlabel = "iteration"
        self._loss_plot = self._subplots[0][0]

        self._training_loss_history = _training_loss_history
        if validation_loss_history is not None:
            validation_loss_mean = np.array(
                validation_loss_history["loss_mean"]
            )
            match validation_loss_mean.ndim:
                case 1:
                    validation_loss_mean = validation_loss_mean[np.newaxis, :]
                case _:
                    pass
            assert len(validation_loss_mean.shape) == 2, (
                "validation_loss_history['loss_mean'] must be a 2D array"
                "with shape (number_of_trainings, iteration_length)"
            )
            validation_loss_std = np.array(validation_loss_history["loss_std"])
            match validation_loss_std.ndim:
                case 1:
                    validation_loss_std = validation_loss_std[np.newaxis, :]
                case _:
                    pass
            assert len(validation_loss_std.shape) == 2, (
                "validation_loss_history['loss_std'] must be a 2D array"
                "with shape (number_of_trainings, iteration_length)"
            )
            _validation_loss_history: Optional[Dict[str, NDArray]] = dict(
                iteration=np.array(validation_loss_history["iteration"]),
                loss_mean=validation_loss_mean,
                loss_std=validation_loss_std,
            )
        else:
            _validation_loss_history = None
        self._validation_loss_history = _validation_loss_history
        self._scaling = scaling

    @property
    def loss_plot_ax(self) -> Axes:
        return self._loss_plot

    def plot(self) -> Self:
        if self._batch_size == 1:
            self._plot_training_trajectory()
        else:
            self._plot_statistic_training_trajectory()
        super().plot()
        return self

    def _plot_training_trajectory(self) -> None:
        training_loss_trajectory = np.array(
            self._training_loss_history["loss"]
        )
        self._plot_signal_trajectory(
            self._loss_plot,
            training_loss_trajectory[0] * self._scaling,
            ylabel="loss",
            label="training",
            zorder=1,
        )
        if self._validation_loss_history is not None:
            self._loss_plot.errorbar(
                self._validation_loss_history["iteration"],
                self._validation_loss_history["loss_mean"][0, :]
                * self._scaling,
                self._validation_loss_history["loss_std"][0, :]
                * self._scaling,
                label="validation",
                fmt=".",
                capsize=3,
                color="C1",
                zorder=2,
            )
        self._loss_plot.legend()
        if self._iteration_log_scale:
            self._loss_plot.set_xscale("log")

    def _plot_statistic_training_trajectory(self) -> None:
        pass
