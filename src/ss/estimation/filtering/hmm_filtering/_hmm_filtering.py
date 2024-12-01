from typing import Optional, Self, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ss.estimation.filtering import Filter
from ss.system.markov import HiddenMarkovModel, one_hot_decoding
from ss.tool.figure import TimeTrajectoryFigure


class HiddenMarkovModelFilter(Filter):
    def __init__(
        self,
        system: HiddenMarkovModel,
        initial_distribution: Optional[ArrayLike] = None,
    ) -> None:
        assert issubclass(type(system), HiddenMarkovModel), (
            f"system must be an instance of HiddenMarkovModel or its subclasses. "
            f"system given is an instance of {type(system)}."
        )
        self._system = system
        super().__init__(
            state_dim=self._system.state_dim,
            observation_dim=self._system.observation_dim,
            number_of_systems=self._system.number_of_systems,
        )

        if initial_distribution is None:
            initial_distribution = np.ones(self._state_dim) / self._state_dim
        initial_distribution = np.array(initial_distribution, dtype=np.float64)
        assert initial_distribution.shape[0] == self._state_dim, (
            f"initial_distribution must be in the shape of {(self._state_dim,) = }. "
            f"initial_distribution given has the shape of {initial_distribution.shape}."
        )
        self._estimated_state[...] = initial_distribution[np.newaxis, :]

    def _compute_estimation_process(self) -> NDArray[np.float64]:
        estimation_process: NDArray[np.float64] = self._estimation_process(
            estimated_state=self._estimated_state,
            observation=self._observation_history[:, :, 0],
            transition_probability_matrix=self._system.transition_probability_matrix,
            emission_probability_matrix=self._system.emission_probability_matrix,
        )
        return estimation_process

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _estimation_process(
        estimated_state: NDArray[np.float64],
        observation: NDArray[np.float64],
        transition_probability_matrix: NDArray[np.float64],
        emission_probability_matrix: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        number_of_systems: int = estimated_state.shape[0]

        # prediction step based on model process (predicted probability)
        estimated_state[...] = estimated_state @ transition_probability_matrix

        # update step based on observation (unnormalized conditional probability)
        # the transpose operation is for the purpose of the multi-system case
        estimated_state[...] = (
            estimated_state
            * emission_probability_matrix[:, one_hot_decoding(observation)].T
        )

        # normalization step (conditional probability)
        for i in range(number_of_systems):
            estimated_state[i, :] /= np.sum(estimated_state[i, :])

        return estimated_state


class HiddenMarkovModelFilterFigure(TimeTrajectoryFigure):
    def __init__(
        self,
        time_trajectory: ArrayLike,
        observation_trajectory: ArrayLike,
        estimated_state_trajectory: ArrayLike,
        fig_size: Tuple = (12, 8),
        fig_title: Optional[str] = None,
        fig_layout: Tuple = (2, 1),
    ) -> None:
        observation_trajectory = np.array(
            observation_trajectory, dtype=np.int64
        )
        match observation_trajectory.ndim:
            case 1:
                observation_trajectory = observation_trajectory[np.newaxis, :]
            case _:
                pass
        assert observation_trajectory.ndim == 2, (
            f"observation_value_trajectory must be in the shape of (state_dim, time_horizon). "
            f"observation_value_trajectory given has the shape of {observation_trajectory.shape}."
        )

        estimated_state_trajectory = np.array(
            estimated_state_trajectory, dtype=np.float64
        )
        match estimated_state_trajectory.ndim:
            case 1:
                estimated_state_trajectory = estimated_state_trajectory[
                    np.newaxis, :
                ]
            case _:
                pass
        assert estimated_state_trajectory.ndim == 2, (
            f"estimated_state_trajectory must be in the shape of (state_dim, time_horizon). "
            f"estimated_state_trajectory given has the shape of {estimated_state_trajectory.shape}."
        )

        super().__init__(
            time_trajectory=time_trajectory,
            fig_size=fig_size,
            fig_title=(
                "Hidden Markov Model Filter" if fig_title is None else fig_title
            ),
            fig_layout=fig_layout,
        )
        assert estimated_state_trajectory.shape[1] == self._time_length, (
            f"estimated_state_trajectory must have the same time horizon as time_trajectory. "
            f"estimated_state_trajectory has the time horizon of {estimated_state_trajectory.shape[1]} "
            f"while time_trajectory has the time horizon of {self._time_length}."
        )
        assert observation_trajectory.shape[1] == self._time_length, (
            f"observation_value_trajectory must have the same time horizon as time_trajectory. "
            f"observation_value_trajectory has the time horizon of {observation_trajectory.shape[1]} "
            f"while time_trajectory has the time horizon of {self._time_length}."
        )

        self._observation_value_trajectory = observation_trajectory
        self._estimated_state_trajectory = estimated_state_trajectory
        self._observation_subplot = self._subplots[0][0]
        self._estimated_state_subplot = self._subplots[1][0]

    def plot(self) -> Self:
        self._plot_probability_flow(
            self._estimated_state_subplot,
            self._estimated_state_trajectory,
        )
        self._estimated_state_subplot.set_ylabel("Estimated State Probability")
        self._plot_probability_flow(
            self._observation_subplot,
            self._observation_value_trajectory,
        )
        self._observation_subplot.set_ylabel("Observation")
        self._sup_xlabel = "Time Step"
        super().plot()

        self._create_color_bar()
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
            label="probability",
        )
        return color_bar
