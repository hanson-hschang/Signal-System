from typing import Dict, Optional, Self, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from numpy.typing import NDArray

from ss import figure as Figure
from ss.system.markov import one_hot_encoding

from ..utility import FilterResultTrajectory


def add_loss_line(
    ax: Axes,
    loss: float,
    text: str = "loss: {:.3f}",
    log_base: float = np.e,
    arrowhead_x_offset_ratio: float = 0.05,
    text_offset: tuple[float, float] = (64, 32),
    text_coordinates: str = "offset pixels",
) -> None:
    ax.axhline(y=loss, color="black", linestyle="--")
    bbox = dict(boxstyle="round", fc="0.8")
    arrowprops = dict(
        arrowstyle="->",
        connectionstyle="angle,angleA=0,angleB=90,rad=10",
    )
    xlim_min, xlim_max = ax.get_xlim()
    xlim_range = xlim_max - xlim_min
    text = text.replace("{:.3f}", "{:.3f} ({:.1f}% accurate)")
    ax.annotate(
        text.format(loss, np.power(log_base, -loss) * 100),
        (xlim_min + arrowhead_x_offset_ratio * xlim_range, loss),
        xytext=text_offset,
        textcoords=text_coordinates,
        bbox=bbox,
        arrowprops=arrowprops,
    )


def update_loss_ylim(
    ax: Axes, min_max_value: Tuple[float, float], margin: float = 0.1
) -> None:
    min_value = min(min_max_value)
    max_value = max(min_max_value)
    range_value = max_value - min_value
    ax.set_ylim(
        bottom=min_value - margin * range_value,
        top=max_value + margin * range_value,
    )


class FilterResultFigure(Figure.SequenceTrajectoryFigure):
    def __init__(
        self,
        time_trajectory: NDArray,
        observation_trajectory: NDArray,
        filter_result_trajectory_dict: Dict[str, FilterResultTrajectory],
        loss_scaling: float = 1.0,
        fig_size: Tuple = (12, 8),
        fig_title: Optional[str] = None,
    ) -> None:
        filter_names = list(filter_result_trajectory_dict.keys())
        discrete_observation_dim = filter_result_trajectory_dict[
            filter_names[0]
        ].estimated_next_observation_probability.shape[0]
        self._number_of_filters = len(filter_names)
        self._filter_result_trajectory_dict = filter_result_trajectory_dict
        self._loss_scaling = loss_scaling
        basis = np.identity(discrete_observation_dim)
        self._observation_trajectory = one_hot_encoding(
            observation_trajectory[0, :-1], basis
        ).T
        self._target_trajectory = one_hot_encoding(
            observation_trajectory[0, 1:], basis
        ).T
        super().__init__(
            sequence_trajectory=time_trajectory[:-1],
            number_of_systems=1,
            fig_size=fig_size,
            fig_title=fig_title,
            fig_layout=(self._number_of_filters + 3, 1),
        )
        self._sup_xlabel = "time step"
        self._loss_subplot = self._subplots[-1][0]
        self._observation_subplot = self._subplots[0][0]
        self._observation_subplot.sharex(self._loss_subplot)
        self._target_subplot = self._subplots[1][0]
        self._target_subplot.sharex(self._loss_subplot)
        self._result_subplots: Dict[str, Axes] = {}
        for i, filter_name in enumerate(filter_result_trajectory_dict.keys()):
            self._result_subplots[filter_name] = self._subplots[i + 2][0]
            self._result_subplots[filter_name].sharex(self._loss_subplot)

    def plot(self) -> Self:
        self._plot_probability_flow(
            ax=self._observation_subplot,
            probability_trajectory=self._observation_trajectory,
            ylabel="observation",
        )
        self._plot_probability_flow(
            ax=self._target_subplot,
            probability_trajectory=self._target_trajectory,
            ylabel="next observation",
        )
        last_horizon_for_loss_mean = int(np.sqrt(self._sequence_length))
        for (
            filter_name,
            filter_result,
        ) in self._filter_result_trajectory_dict.items():
            label_name = filter_name.replace("_", " ")

            self._plot_signal_trajectory(
                ax=self._loss_subplot,
                signal_trajectory=filter_result.loss * self._loss_scaling,
                ylabel="loss",
                label=label_name
                + f" (loss: {np.mean(filter_result.loss[-last_horizon_for_loss_mean:]):.2f})",
            )
            self._plot_probability_flow(
                ax=self._result_subplots[filter_name],
                probability_trajectory=filter_result.estimated_next_observation_probability,
                ylabel=label_name,
            )
        self._loss_subplot.legend()
        super().plot()

        self._create_color_bar()
        self._adjust_subplots_location()
        return self

    def _create_color_bar(self) -> Colorbar:
        ax: Axes = self._fig.subplots(1, 1)
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.patch.set_alpha(0.0)

        color_bar: Colorbar = self._fig.colorbar(
            mappable=plt.cm.ScalarMappable(
                cmap=self._color_map,
                norm=Normalize(vmin=0, vmax=1),
            ),
            ax=ax,
            label="\nProbability Color Bar",
        )
        return color_bar

    def _adjust_subplots_location(self) -> None:
        axes = [
            self._observation_subplot,
            self._target_subplot,
            *self._result_subplots.values(),
            self._loss_subplot,
        ]

        space = 1 / 1.2 / len(axes) / 5
        height = 3 * space
        for k, ax in enumerate(axes):
            ax.set_position(
                (
                    0.05,
                    1.1 / 1.2 - (k + 1) * (2 * space + height) + space,
                    0.75,
                    height,
                )
            )
