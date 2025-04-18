from typing import Optional, Tuple

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
        max_horizon_of_observation_history: int,
        initial_distribution: Optional[ArrayLike] = None,
        terminal_estimation: Optional[ArrayLike] = None,
    ) -> None:
        self._system = system
        self._horizon_of_observation_history = 0
        self._max_horizon_of_observation_history = (
            max_horizon_of_observation_history
        )
        self._discrete_state_dim = self._system.discrete_state_dim
        self._observation_dim = self._system.observation_dim
        assert self._observation_dim == 1, (
            f"observation_dim must be 1. "
            f"observation_dim given is {self._observation_dim}."
        )
        self._discrete_observation_dim = self._system.discrete_observation_dim

        self._initial_distribution = (
            np.full(self._discrete_state_dim, 1 / self._discrete_state_dim)
            if initial_distribution is None
            else np.array(initial_distribution)
        )
        self._terminal_dual_function = (
            np.identity(self._system.discrete_state_dim)
            if terminal_estimation is None
            else np.array(terminal_estimation)
        )

        self._number_of_systems = self._system.number_of_systems
        self._observation_history = np.full(
            (
                self._number_of_systems,
                self._observation_dim,
                self._max_horizon_of_observation_history,
            ),
            np.nan,
            dtype=np.float64,
        )
        self._emission_history = np.full(
            (
                self._number_of_systems,
                self._discrete_state_dim,
                self._max_horizon_of_observation_history,
            ),
            np.nan,
            dtype=np.float64,
        )
        self._likelihood_history = np.empty(
            (
                self._number_of_systems,
                self._discrete_state_dim,
                self._max_horizon_of_observation_history,
            ),
            dtype=np.float64,
        )
        for k in range(self._max_horizon_of_observation_history):
            for i in range(self._number_of_systems):
                self._likelihood_history[i, :, k] = (
                    self._initial_distribution.copy()
                )

        self._estimated_distribution_history = np.empty(
            (
                self._number_of_systems,
                self._discrete_state_dim,
                self._max_horizon_of_observation_history,
            ),
            dtype=np.float64,
        )

    observation_history = MultiSystemNdArrayReadOnlyDescriptor(
        "_number_of_systems",
        "_observation_dim",
        "_horizon_of_observation_history",
    )
    emission_difference_history = MultiSystemNdArrayReadOnlyDescriptor(
        "_number_of_systems",
        "_discrete_state_dim",
        "_horizon_of_observation_history",
    )
    likelihood_history = MultiSystemNdArrayReadOnlyDescriptor(
        "_number_of_systems",
        "_discrete_state_dim",
        "_horizon_of_observation_history",
    )
    estimated_distribution_history = MultiSystemNdArrayReadOnlyDescriptor(
        "_number_of_systems",
        "_discrete_state_dim",
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
        observation = np.array(observation, dtype=np.int64)
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
            emission_matrix=self._system.emission_matrix,
            observation_history=self._observation_history,
            emission_history=self._emission_history,
            likelihood_history=self._likelihood_history,
        )
        self._horizon_of_observation_history = min(
            self._horizon_of_observation_history + 1,
            self._max_horizon_of_observation_history,
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _update_observation(
        observation: NDArray[np.int64],
        emission_matrix: NDArray[np.float64],
        observation_history: NDArray[np.float64],
        emission_history: NDArray[np.float64],
        likelihood_history: NDArray[np.float64],
    ) -> None:
        number_of_systems, _, _ = observation_history.shape
        discrete_state_dim, _ = emission_matrix.shape

        # Move the observation, emission, and estimated distribution history one step into the past
        for i in range(number_of_systems):
            # numba v0.61.0 does not support np.roll with axis argument
            observation_history[i, 0, :] = np.roll(
                observation_history[i, 0, :], shift=-1
            )
            for d in range(discrete_state_dim):
                emission_history[i, d, :] = np.roll(
                    emission_history[i, d, :], shift=-1
                )
                likelihood_history[i, d, :] = np.roll(
                    likelihood_history[i, d, :], shift=-1
                )

        # Update the most recent history based on the new observation
        emission_column = emission_matrix[
            :, observation[:, 0]
        ].T  # (number_of_systems, discrete_state_dim)
        emission_history[:, :, -1] = 2 * emission_column - 1
        for i in range(number_of_systems):
            # numba v0.61.0 does not support np.sum with axis argument
            likelihood_history[i, :, -1] = emission_column[i, :] / np.sum(
                emission_column[i, :]
            )
        observation_history[:, :, -1] = observation

    def estimate(
        self, number_of_iterations: int = 1
    ) -> Tuple[NDArray, NDArray]:
        """
        Estimate the distribution of the system.

        Parameters
        ----------
        number_of_iterations : int
            The number of iterations to run the estimation.

        Returns
        -------
        estimated_distribution : NDArray
            The estimated distribution of the system.
        """
        number_of_dual_function = self._terminal_dual_function.shape[1]
        control_history = np.empty(
            (
                number_of_iterations,
                self._number_of_systems,
                number_of_dual_function,
                self._max_horizon_of_observation_history,
            ),
            dtype=np.float64,
        )
        estimated_distribution_history = np.empty(
            (
                number_of_iterations + 1,
                self._number_of_systems,
                self._discrete_state_dim,
                self._max_horizon_of_observation_history,
            ),
            dtype=np.float64,
        )
        estimated_distribution_history[0, ...] = (
            self._likelihood_history.copy()
        )
        for i in range(number_of_iterations):
            (
                control_history[i, ...],
                estimated_distribution_history[i + 1, ...],
            ) = self._estimate(estimated_distribution_history[i, ...])

        self._estimated_distribution_history[...] = (
            estimated_distribution_history[-1, ...]
        )
        if number_of_iterations == 1:
            return (
                control_history[0, ...],
                estimated_distribution_history[0, ...],
            )
        return control_history, estimated_distribution_history

    def _estimate(
        self, likelihood_history: NDArray
    ) -> Tuple[NDArray, NDArray]:
        number_of_dual_function = self._terminal_dual_function.shape[1]
        dual_function_history = np.empty(
            (
                self._number_of_systems,
                self._discrete_state_dim,
                number_of_dual_function,
                self._max_horizon_of_observation_history + 1,
            ),
            dtype=np.float64,
        )
        dual_function_history[:, :, :, -1] = np.repeat(
            self._terminal_dual_function[np.newaxis, ...],
            self._number_of_systems,
            axis=0,
        )

        control_history = np.empty(
            (
                self._number_of_systems,
                number_of_dual_function,
                self._max_horizon_of_observation_history,
            ),
            dtype=np.float64,
        )

        estimated_distribution_history = np.empty(
            (
                self._number_of_systems,
                self._discrete_state_dim,
                self._max_horizon_of_observation_history,
            ),
            dtype=np.float64,
        )

        # Backward in time
        for k in range(self._horizon_of_observation_history):
            dual_function = dual_function_history[
                :, :, :, -1 - k
            ]  # (number_of_systems, discrete_state_dim, number_of_dual_functions)
            emission = self._emission_history[
                :, :, -1 - k
            ]  # (number_of_systems, discrete_state_dim)

            # estimated_distribution = (
            #     self._initial_distribution
            #     if k == (self._horizon_of_observation_history - 1)
            #     else self._estimated_distribution_history[0, :, -1 - k - 1]
            # )
            past_estimated_distribution = (
                np.repeat(
                    self._initial_distribution[np.newaxis, :],
                    self._number_of_systems,
                    axis=0,
                )
                if k == (self._horizon_of_observation_history - 1)
                else likelihood_history[:, :, -1 - k - 1]
            )  # (number_of_systems, discrete_state_dim)

            past_dual_function = self._compute_past_dual_function(
                self._system.transition_matrix,
                dual_function,
            )

            # Compute the control
            control = self._compute_control(
                past_dual_function,
                emission,
                past_estimated_distribution,
            )  # (number_of_systems, number_of_dual_functions)

            # Update the dual function
            dual_function_history[:, :, :, -1 - k - 1] = self._backward_step(
                past_dual_function,
                emission,
                control,
            )

            control_history[:, :, -1 - k] = control

        # Compute the initial estimator
        estimator = self._compute_initial_estimator(
            self._initial_distribution,
            dual_function_history[..., -1 - k - 1],
        )  # (number_of_systems, number_of_dual_function)

        # estimator = (
        #     self._initial_distribution @ dual_function_history[0, :, :, 0]
        # )

        # Update the estimated distribution
        for k in range(
            self._max_horizon_of_observation_history
            - self._horizon_of_observation_history,
            self._max_horizon_of_observation_history,
        ):
            estimator = self._compute_estimator(
                estimator,
                control_history[:, :, k],
            )
            # estimator = estimator - control_history[0, :, k]

            estimated_distribution_history[:, :, k] = (
                self._update_estimated_distribution(
                    dual_function_history[:, :, :, k + 1],
                    estimator,
                )
            )
            # except np.linalg.LinAlgError:
            #     if np.any(estimated_distribution_history==np.nan):
            #         print("estimated_distribution_history contains NaN")
            #     if np.any(dual_function_history == np.nan):
            #         print("dual_function_history contains NaN")
            #     if np.any(estimator == np.nan):
            #         print("estimator contains NaN")
            #     print(dual_function_history[:, :, :, k + 1])
            #     print(estimator)
            #     quit()
        return control_history, estimated_distribution_history

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _compute_past_dual_function(
        transition_matrix: NDArray[np.float64],
        dual_function: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        number_of_systems = dual_function.shape[0]
        past_dual_function = np.empty_like(dual_function)
        for i in range(number_of_systems):
            # copy is used to avoid the following performance warning
            # NumbaPerformanceWarning: '@' is faster on contiguous arrays
            past_dual_function[i, :, :] = (
                transition_matrix @ dual_function[i, :, :].copy()
            )
        return past_dual_function

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _compute_control(
        past_dual_function: NDArray[np.float64],
        emission: NDArray[np.float64],
        estimated_distribution: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        number_of_systems, _, number_of_dual_functions = (
            past_dual_function.shape
        )
        control = np.empty((number_of_systems, number_of_dual_functions))
        for i in range(number_of_systems):

            expected_emission = np.sum(
                estimated_distribution[i, :] * emission[i, :]
            )
            denominator = 1 - (expected_emission**2)

            if denominator == 0:
                control[i, :] = 0
                continue

            for d in range(number_of_dual_functions):
                # control[i, d] = -(
                #     np.sum(
                #         estimated_distribution[i, :] * (
                #             past_dual_function[i, :, d] * (
                #                 emission[i, :] - expected_emission
                #             )
                #         )
                #     )
                # ) / denominator
                control[i, d] = (
                    (
                        np.sum(
                            estimated_distribution[i, :]
                            * past_dual_function[i, :, d]
                        )
                        * expected_emission
                    )
                    - np.sum(
                        estimated_distribution[i, :]
                        * (past_dual_function[i, :, d] * emission[i, :])
                    )
                ) / denominator
        return control

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _backward_step(
        past_dual_function: NDArray[np.float64],
        emission: NDArray[np.float64],
        control: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        number_of_systems = past_dual_function.shape[0]
        updated_past_dual_function = np.empty_like(past_dual_function)
        for i in range(number_of_systems):
            updated_past_dual_function[i, :, :] = past_dual_function[
                i, :, :
            ] + np.outer(emission[i, :], control[i, :])
        # result = past_dual_function + np.outer(
        #     emission, control
        # )
        return updated_past_dual_function

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _compute_initial_estimator(
        initial_distribution: NDArray[np.float64],
        dual_function: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        number_of_systems, _, number_of_dual_functions = dual_function.shape
        initial_estimator = np.empty(
            (number_of_systems, number_of_dual_functions)
        )
        for i in range(number_of_systems):
            for d in range(number_of_dual_functions):
                initial_estimator[i, d] = np.sum(
                    initial_distribution * dual_function[i, :, d]
                )
            # initial_estimator[i, :] = (
            #     initial_distribution @ dual_function[i, :, :]
            # )
        return initial_estimator

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _compute_estimator(
        past_estimator: NDArray[np.float64],
        control: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return past_estimator - control

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _update_estimated_distribution(
        dual_function: NDArray[np.float64],
        estimator: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        number_of_systems, discrete_state_dim, _ = dual_function.shape
        estimated_distribution = np.empty(
            (number_of_systems, discrete_state_dim)
        )
        for i in range(number_of_systems):
            result, _, _, _ = np.linalg.lstsq(
                dual_function[i, :, :].T,
                estimator[i, :],
            )  # (discrete_state_dim,)
            result = np.maximum(result, 0)
            estimated_distribution[i, :] = result / np.sum(result)
        # result = np.linalg.solve(dual_function.T, estimator)
        # result = np.maximum(result, 0)
        # result /= np.sum(result)
        return estimated_distribution
