from typing import Optional

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ss.system.markov import HiddenMarkovModel
from ss.utility.descriptor import (
    MultiSystemNDArrayDescriptor,
    MultiSystemNdArrayReadOnlyDescriptor,
    ReadOnlyDescriptor,
)


class DualHmmFilter:
    def __init__(
        self,
        system: HiddenMarkovModel,
        time_horizon: int,
        initial_distribution: Optional[ArrayLike] = None,
        terminal_estimation: Optional[ArrayLike] = None,
    ) -> None:
        self._system = system
        self._time_horizon = time_horizon
        self._discrete_state_dim = self._system.discrete_state_dim
        self._observation_dim = self._system.observation_dim
        self._discrete_observation_dim = self._system.discrete_observation_dim

        self._initial_distribution = (
            np.ones(self._discrete_state_dim) / self._discrete_state_dim
            if initial_distribution is None
            else np.array(initial_distribution)
        )
        self._terminal_dual_function = (
            np.identity(self._system.discrete_state_dim)
            if terminal_estimation is None
            else np.array(terminal_estimation)
        )

        self._number_of_systems = self._system.number_of_systems
        self._observation_history = np.zeros(
            (
                self._number_of_systems,
                self._observation_dim,
                self._time_horizon,
            ),
            dtype=np.float64,
        )
        self._emission_difference_history = np.zeros(
            (
                self._number_of_systems,
                self._discrete_state_dim,
                self._time_horizon,
            ),
            dtype=np.float64,
        )
        self._estimated_distribution_history = np.zeros(
            (
                self._number_of_systems,
                self._discrete_state_dim,
                self._time_horizon,
            ),
            dtype=np.float64,
        )
        for k in range(self._time_horizon):
            for n in range(self._number_of_systems):
                self._estimated_distribution_history[n, :, k] = (
                    self._initial_distribution.copy()
                )

        # self._control_history = np.zeros(
        #     (
        #         self._number_of_systems,
        #         self._observation_dim,
        #         self._time_horizon,
        #     ),
        #     dtype=np.float64,
        # )

    observation_history = MultiSystemNdArrayReadOnlyDescriptor(
        "_number_of_systems",
        "_observation_dim",
        "_time_horizon",
    )
    emission_difference_history = MultiSystemNdArrayReadOnlyDescriptor(
        "_number_of_systems",
        "_discrete_state_dim",
        "_time_horizon",
    )
    estimated_distribution_history = MultiSystemNdArrayReadOnlyDescriptor(
        "_number_of_systems",
        "_discrete_state_dim",
        "_time_horizon",
    )

    def update(self, observation: ArrayLike) -> None:
        """
        Update the observation history with the given observation.

        Parameters
        ----------
        observation : ArrayLike
            shape = (number_of_systems, observation_dim)
            The observation to be updated.
        """
        observation = np.array(observation, dtype=np.float64)
        if observation.ndim == 1:
            observation = observation[np.newaxis, :]
        assert observation.shape == (
            self._number_of_systems,
            self._observation_dim,
        ), (
            f"observation must be in the shape of {(self._number_of_systems, self._observation_dim) = }. "
            f"observation given has the shape of {observation.shape}."
        )
        self._update_observation(
            observation=observation,
            observation_history=self._observation_history,
            emission_difference_history=self._emission_difference_history,
            emission_matrix=self._system.emission_matrix,
            estimated_distribution_history=self._estimated_distribution_history,
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _update_observation(
        observation: NDArray[np.float64],
        observation_history: NDArray[np.float64],
        emission_difference_history: NDArray[np.float64],
        emission_matrix: NDArray[np.float64],
        estimated_distribution_history: NDArray[np.float64],
    ) -> None:
        number_of_systems, observation_dim, _ = observation_history.shape
        discrete_state_dim, _ = emission_matrix.shape

        # Move the observation, emission difference, and estimated distribution history one step into the past
        for i in range(number_of_systems):
            for m in range(observation_dim):
                observation_history[i, m, :] = np.roll(
                    observation_history[i, m, :], -1
                )
            for d in range(discrete_state_dim):
                emission_difference_history[i, d, :] = np.roll(
                    emission_difference_history[i, d, :], -1
                )
                estimated_distribution_history[i, d, :] = np.roll(
                    estimated_distribution_history[i, d, :], -1
                )
            emission = emission_matrix[:, int(observation[i, 0])]
            emission_difference_history[i, :, -1] = 2 * emission - 1
            estimated_distribution_history[i, :, -1] = emission / np.sum(
                emission
            )

        # Update the most recent history with the new observation
        observation_history[:, :, -1] = observation
        # emission_difference_history[:, :, 0] = 2 * emission_matrix[:, observation[:, 0].astype(np.int64)] - 1

    def estimate(self) -> NDArray:
        dual_function_history = np.empty(
            (
                self._number_of_systems,
                self._discrete_state_dim,
                self._discrete_state_dim,
                self._time_horizon + 1,
            ),
            dtype=np.float64,
        )
        control_history = np.empty(
            (
                self._number_of_systems,
                self._discrete_state_dim,
                self._time_horizon,
            ),
            dtype=np.float64,
        )

        # Backward in time
        dual_function_history[0, :, :, -1] = (
            self._terminal_dual_function.copy()
        )
        for k in range(self._time_horizon):
            dual_function = dual_function_history[0, :, :, -1 - k]
            emission_difference = self._emission_difference_history[
                0, :, -1 - k
            ]
            estimated_distribution = (
                self._initial_distribution
                if k == (self._time_horizon - 1)
                else self._estimated_distribution_history[0, :, -1 - k - 1]
            )

            control = self._compute_control(
                self._system.transition_matrix @ dual_function,
                emission_difference,
                estimated_distribution,
            )
            dual_function_history[0, :, :, -1 - k - 1] = self._backward_step(
                self._system.transition_matrix,
                dual_function,
                emission_difference,
                control,
            )

            control_history[0, :, -1 - k] = control

        estimator = (
            self._initial_distribution @ dual_function_history[0, :, :, 0]
        )

        # Update the estimated distribution
        for k in range(self._time_horizon):
            estimator = estimator - control_history[0, :, k]
            self._estimated_distribution_history[0, :, k] = (
                self._update_estimated_distribution(
                    dual_function_history[0, :, :, k + 1],
                    estimator,
                )
            )

        return self._estimated_distribution_history[0, :, -1]

    def _compute_control(
        self,
        dual_function: NDArray[np.float64],
        emission_difference: NDArray[np.float64],
        estimated_distribution: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        denominator = 1 - (estimated_distribution @ emission_difference) ** 2
        if denominator == 0:
            return np.zeros_like(estimated_distribution)
        expected_emission = estimated_distribution @ emission_difference
        _, dual_function_dim = dual_function.shape
        control = np.zeros(dual_function_dim)
        for d in range(dual_function_dim):
            control[d] = (
                -estimated_distribution
                @ (
                    dual_function[:, d]
                    * (emission_difference - expected_emission)
                )
                / denominator
            )
        return control

    def _backward_step(
        self,
        transition_matrix: NDArray[np.float64],
        dual_function: NDArray[np.float64],
        emission_difference: NDArray[np.float64],
        control: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        result = transition_matrix @ dual_function + np.outer(
            emission_difference, control
        )
        return result

    def _compute_estimator(
        self,
        estimator: NDArray[np.float64],
        control: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return estimator - control

    def _update_estimated_distribution(
        self,
        dual_function: NDArray[np.float64],
        estimator: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        result = np.linalg.solve(dual_function.T, estimator)
        result = np.maximum(result, 0)
        result /= np.sum(result)
        return result
