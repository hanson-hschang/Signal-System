from typing import Optional, Self, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from numpy.typing import ArrayLike, NDArray

from ss.utility.assertion.validator import Validator
from ss.utility.figure import SequenceTrajectoryFigure


class HiddenMarkovModelFilterFigure(SequenceTrajectoryFigure):

    class _SignalTrajectoryValidator(Validator):
        def __init__(
            self,
            signal_trajectory: Optional[ArrayLike],
            time_length: int,
            signal_name: str = "signal_trajectory",
        ) -> None:
            super().__init__()
            if signal_trajectory is None:
                signal_trajectory = np.full((1, time_length), np.nan)
            self._signal_trajectory = np.array(
                signal_trajectory, dtype=np.float64
            )
            self._time_length = time_length
            self._signal_name = signal_name
            self._validate_shape()

        def _validate_shape(self) -> None:
            match self._signal_trajectory.ndim:
                case 1:
                    self._signal_trajectory = self._signal_trajectory[
                        np.newaxis, :
                    ]
                case _:
                    pass
            assert self._signal_trajectory.ndim == 2, (
                f"{self._signal_name} must be in the shape of (signal_dim, time_horizon). "
                f"{self._signal_name} given has the shape of {self._signal_trajectory.shape}."
            )
            assert self._signal_trajectory.shape[1] == self._time_length, (
                f"{self._signal_name} must have the same time horizon as time_trajectory. "
                f"{self._signal_name} has the time horizon of {self._signal_trajectory.shape[1]} "
                f"while time_trajectory has the time horizon of {self._time_length}."
            )

        def get_trajectory(self) -> NDArray[np.float64]:
            return self._signal_trajectory

        @staticmethod
        def has_value(signal_trajectory: NDArray[np.float64]) -> bool:
            return not bool(np.all(np.isnan(signal_trajectory)))

    def __init__(
        self,
        time_trajectory: ArrayLike,
        state_trajectory: ArrayLike,
        observation_trajectory: ArrayLike,
        estimated_state_trajectory: ArrayLike,
        estimated_function_value_trajectory: Optional[ArrayLike] = None,
        fig_size: Tuple = (12, 8),
        fig_title: Optional[str] = None,
    ) -> None:

        time_length = np.array(time_trajectory).shape[0]
        self._state_trajectory = self._SignalTrajectoryValidator(
            signal_trajectory=state_trajectory,
            time_length=time_length,
            signal_name="state_trajectory",
        ).get_trajectory()
        self._observation_trajectory = self._SignalTrajectoryValidator(
            signal_trajectory=observation_trajectory,
            time_length=time_length,
            signal_name="observation_trajectory",
        ).get_trajectory()
        self._estimated_state_trajectory = self._SignalTrajectoryValidator(
            signal_trajectory=estimated_state_trajectory,
            time_length=time_length,
            signal_name="estimated_state_trajectory",
        ).get_trajectory()
        self._estimated_function_value_trajectory = (
            self._SignalTrajectoryValidator(
                signal_trajectory=estimated_function_value_trajectory,
                time_length=time_length,
                signal_name="estimated_function_value_trajectory",
            ).get_trajectory()
        )

        fig_layout: Tuple = (3, 1)
        if self._SignalTrajectoryValidator.has_value(
            self._estimated_function_value_trajectory,
        ):
            fig_layout = (4, 1)

        super().__init__(
            sequence_trajectory=time_trajectory,
            fig_size=fig_size,
            fig_title=(
                "Hidden Markov Model Filter" if fig_title is None else fig_title
            ),
            fig_layout=fig_layout,
        )

        self._state_subplot = self._subplots[0][0]
        self._observation_subplot = self._subplots[1][0]
        self._estimated_state_subplot = self._subplots[2][0]
        if self._SignalTrajectoryValidator.has_value(
            self._estimated_function_value_trajectory,
        ):
            self._estimated_observation_subplot = self._subplots[3][0]

    def plot(self) -> Self:
        self._plot_probability_flow(
            self._state_subplot,
            self._state_trajectory,
        )
        self._state_subplot.set_ylabel("State\n")
        self._plot_probability_flow(
            self._observation_subplot,
            self._observation_trajectory,
        )
        self._observation_subplot.set_ylabel("Observation\n")
        self._plot_probability_flow(
            self._estimated_state_subplot,
            self._estimated_state_trajectory,
        )
        self._estimated_state_subplot.set_ylabel(
            "Probability of\nState Estimation\n"
        )
        if self._SignalTrajectoryValidator.has_value(
            self._estimated_function_value_trajectory,
        ):
            self._plot_probability_flow(
                self._estimated_observation_subplot,
                self._estimated_function_value_trajectory,
            )
            self._estimated_observation_subplot.set_ylabel(
                "Probability of\nFunction Value\nEstimation\n"
            )

        self._sup_xlabel = "Time Step"
        super().plot()

        self._create_color_bar()
        self._adjust_subplots_location()
        return self

    def _create_color_bar(self) -> Colorbar:
        ax = self._fig.subplots(1, 1)
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.patch.set_alpha(0.0)

        color_bar = self._fig.colorbar(
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
            self._state_subplot,
            self._observation_subplot,
            self._estimated_state_subplot,
        ]
        if self._SignalTrajectoryValidator.has_value(
            self._estimated_function_value_trajectory,
        ):
            axes.append(self._estimated_observation_subplot)
        space = 1 / 1.2 / len(axes) / 5
        height = 3 * space
        for k, ax in enumerate(axes):
            ax.set_position(
                (
                    0.1,
                    1.1 / 1.2 - (k + 1) * (2 * space + height) + space,
                    0.7,
                    height,
                )
            )
