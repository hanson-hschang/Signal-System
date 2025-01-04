from typing import Callable, Optional, Self, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ss.estimation import EstimatorCallback
from ss.estimation.filtering import Filter
from ss.system.markov import HiddenMarkovModel, one_hot_decoding
from ss.utility.assertion.validator import Validator
from ss.utility.descriptor import MultiSystemNDArrayReadOnlyDescriptor
from ss.utility.figure import SequenceTrajectoryFigure


class HiddenMarkovModelFilter(Filter):
    def __init__(
        self,
        system: HiddenMarkovModel,
        initial_distribution: Optional[ArrayLike] = None,
        estimation_model: Optional[Callable] = None,
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

        if estimation_model is None:

            @njit(cache=True)  # type: ignore
            def _estimation_model(
                estimated_state: NDArray[np.float64],
                number_of_systems: int = self._system.number_of_systems,
            ) -> NDArray[np.float64]:
                return np.full((number_of_systems, 1), np.nan)

            estimation_model = _estimation_model
        self._estimation_model = estimation_model

        self._estimated_function_value: NDArray[np.float64] = (
            self._estimation_model(self._estimated_state)
        )
        self._function_value_dim = self._estimated_function_value.shape[1]

    estimated_function_value = MultiSystemNDArrayReadOnlyDescriptor(
        "_number_of_systems", "_function_value_dim"
    )

    def _compute_estimation_process(self) -> NDArray[np.float64]:
        estimation_process: NDArray[np.float64] = self._estimation_process(
            estimated_state=self._estimated_state,
            observation=self._observation_history[:, :, 0],
            transition_probability_matrix=self._system.transition_probability_matrix,
            emission_probability_matrix=self._system.emission_probability_matrix,
        )
        # self._estimated_state will only be updated by estimation_process
        # in the next step in _update method, so the computation of self._estimated_function_value
        # directly use the estimation_process (instead of self._estimated_state) to avoid one step delay
        self._estimated_function_value[...] = self._estimation_model(
            estimation_process
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


class HiddenMarkovModelFilterCallback(EstimatorCallback):
    def __init__(
        self,
        step_skip: int,
        estimator: HiddenMarkovModelFilter,
    ) -> None:
        assert issubclass(type(estimator), HiddenMarkovModelFilter), (
            f"estimator must be an instance of HiddenMarkovModelFilter or its subclasses. "
            f"estimator given is an instance of {type(estimator)}."
        )
        super().__init__(step_skip, estimator)
        self._estimator: HiddenMarkovModelFilter = estimator

    def _record(self, time: float) -> None:
        super()._record(time)
        self._callback_params["estimated_function_value"].append(
            self._estimator.estimated_function_value.copy()
        )


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
        observation_trajectory: ArrayLike,
        estimated_state_trajectory: ArrayLike,
        estimated_function_value_trajectory: Optional[ArrayLike] = None,
        fig_size: Tuple = (12, 8),
        fig_title: Optional[str] = None,
    ) -> None:

        time_length = np.array(time_trajectory).shape[0]
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

        fig_layout: Tuple = (2, 1)
        if self._SignalTrajectoryValidator.has_value(
            self._estimated_function_value_trajectory,
        ):
            fig_layout = (3, 1)

        super().__init__(
            sequence_trajectory=time_trajectory,
            fig_size=fig_size,
            fig_title=(
                "Hidden Markov Model Filter" if fig_title is None else fig_title
            ),
            fig_layout=fig_layout,
        )

        self._observation_subplot = self._subplots[0][0]
        self._estimated_state_subplot = self._subplots[1][0]
        if self._SignalTrajectoryValidator.has_value(
            self._estimated_function_value_trajectory,
        ):
            self._estimated_observation_subplot = self._subplots[2][0]

    def plot(self) -> Self:
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
                "Probability of\nFunction Value Estimation\n"
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
