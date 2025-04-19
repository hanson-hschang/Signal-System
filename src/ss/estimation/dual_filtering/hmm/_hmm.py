from typing import Optional, Tuple

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ss.system.markov import HiddenMarkovModel
from ss.utility.callback import Callback
from ss.utility.descriptor import (
    MultiSystemNDArrayDescriptor,
    MultiSystemNdArrayReadOnlyDescriptor,
    ReadOnlyDescriptor,
)


class DualHmmFilter:
    def __init__(
        self,
        system: HiddenMarkovModel,
        max_horizon: int,
        initial_distribution: Optional[ArrayLike] = None,
        terminal_estimation: Optional[ArrayLike] = None,
    ) -> None:
        self._system = system
        self._horizon = 0
        self._max_horizon = max_horizon + 1
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
        self._number_of_dual_function = self._terminal_dual_function.shape[1]

        self._number_of_systems = self._system.number_of_systems
        self._observation_history = np.full(
            (
                self._number_of_systems,
                self._observation_dim,
                self._max_horizon,
            ),
            np.nan,
            dtype=np.float64,
        )
        self._emission_history = np.full(
            (
                self._number_of_systems,
                self._discrete_state_dim,
                self._max_horizon,
            ),
            np.nan,
            dtype=np.float64,
        )
        self._likelihood_history = np.full(
            (
                self._number_of_systems,
                self._discrete_state_dim,
                self._max_horizon,
            ),
            np.nan,
            dtype=np.float64,
        )
        self._likelihood_history[:, :, -1] = np.repeat(
            self._initial_distribution[np.newaxis, :],
            self._number_of_systems,
            axis=0,
        )
        # for k in range(self._max_horizon_of_observation_history):
        #     for i in range(self._number_of_systems):
        #         self._likelihood_history[i, :, k] = (
        #             self._initial_distribution.copy()
        #         )

        self._estimated_distribution_history = np.full(
            (
                self._number_of_systems,
                self._discrete_state_dim,
                self._max_horizon,
            ),
            np.nan,
            dtype=np.float64,
        )
        self._estimated_distribution_history[:, :, -1] = (
            self._likelihood_history[:, :, -1].copy()
        )

        self._control_history = np.full(
            (
                self._number_of_systems,
                self._number_of_dual_function,
                self._max_horizon,
            ),
            np.nan,
            dtype=np.float64,
        )

    observation_history = MultiSystemNdArrayReadOnlyDescriptor(
        "_number_of_systems",
        "_observation_dim",
        "_max_horizon",
    )
    emission_history = MultiSystemNdArrayReadOnlyDescriptor(
        "_number_of_systems",
        "_discrete_state_dim",
        "_max_horizon",
    )
    likelihood_history = MultiSystemNdArrayReadOnlyDescriptor(
        "_number_of_systems",
        "_discrete_state_dim",
        "_max_horizon",
    )
    estimated_distribution_history = MultiSystemNdArrayReadOnlyDescriptor(
        "_number_of_systems",
        "_discrete_state_dim",
        "_max_horizon",
    )
    control_history = MultiSystemNdArrayReadOnlyDescriptor(
        "_number_of_systems",
        "_number_of_dual_function",
        "_max_horizon",
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
            initial_distribution=self._initial_distribution,
            observation_history=self._observation_history,
            emission_history=self._emission_history,
            likelihood_history=self._likelihood_history,
            estimated_distribution_history=self._estimated_distribution_history,
        )
        self._horizon = min(
            self._horizon + 1,
            self._max_horizon - 1,
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _update_observation(
        observation: NDArray[np.int64],
        emission_matrix: NDArray[np.float64],
        initial_distribution: NDArray[np.float64],
        observation_history: NDArray[np.float64],
        emission_history: NDArray[np.float64],
        likelihood_history: NDArray[np.float64],
        estimated_distribution_history: NDArray[np.float64],
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
                estimated_distribution_history[i, d, :] = np.roll(
                    estimated_distribution_history[i, d, :], shift=-1
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
            if likelihood_history[i, 0, 0] != np.nan:
                likelihood_history[i, :, 0] = initial_distribution.copy()
            if emission_history[i, 0, 0] != np.nan:
                emission_history[i, :, 0] = np.nan
            if observation_history[i, 0, 0] != np.nan:
                observation_history[i, 0, 0] = np.nan
        estimated_distribution_history[:, :, -1] = likelihood_history[
            :, :, -1
        ].copy()
        observation_history[:, :, -1] = observation.copy()

    def estimate(self, iterations: int = 1) -> Tuple[NDArray, NDArray]:
        """
        Estimate the distribution of the system.

        Parameters
        ----------
        iterations : int
            The number of iterations to run the estimation.

        Returns
        -------
        estimated_distribution : NDArray
            The estimated distribution of the system.
        """
        number_of_dual_functions = self._terminal_dual_function.shape[1]
        control_history = np.empty(
            (
                iterations,
                self._number_of_systems,
                number_of_dual_functions,
                self._horizon + 1,
            ),
        )
        estimated_distribution_history = np.empty(
            (
                iterations + 1,
                self._number_of_systems,
                self._discrete_state_dim,
                self._horizon + 1,
            ),
        )
        estimated_distribution_history[0, ...] = self._likelihood_history[
            ..., -1 - self._horizon :
        ].copy()
        # estimated_distribution_history[0, ...] = self._estimated_distribution_history[
        #     ..., -1 - self._horizon_of_observation_history :
        # ].copy()

        for i in range(iterations):
            (
                control_history[i, ...],
                estimated_distribution_history[i + 1, ...],
            ) = self._estimate(
                estimated_distribution_history[i, ...],
                self._emission_history[..., -1 - self._horizon :],
            )

        self._control_history[..., -1 - self._horizon :] = control_history[
            -1, ...
        ].copy()

        self._estimated_distribution_history[..., -1 - self._horizon :] = (
            estimated_distribution_history[-1, ...].copy()
        )

        if self._horizon == self._max_horizon - 1:
            self._initial_distribution = self._estimated_distribution_history[
                ..., -self._horizon
            ].copy()

        return control_history, estimated_distribution_history

    def _estimate(
        self,
        estimated_distribution_history: NDArray,
        emission_history: NDArray,
    ) -> Tuple[NDArray, NDArray]:

        # control_history, updated_estimated_distribution_history = _terminal_path(
        #     estimated_distribution_history,
        #     emission_history,
        #     self._system.transition_matrix,
        #     self._terminal_dual_function,
        # )

        control_history, updated_estimated_distribution_history = _multi_path(
            estimated_distribution_history,
            emission_history,
            self._system.transition_matrix,
            self._terminal_dual_function,
        )

        # The following code is the single path implementation

        # dual_function_history, control_history = _backward_path(
        #     estimated_distribution_history,
        #     emission_history,
        #     self._system.transition_matrix,
        #     self._terminal_dual_function,
        # )

        # updated_estimated_distribution_history = _forward_path(
        #     dual_function_history,
        #     control_history,
        #     estimated_distribution_history[:, :, 0],
        # )

        # The following code is the multi path implementation

        # updated_estimated_distribution_history = np.empty(
        #     (
        #         self._number_of_systems,
        #         self._discrete_state_dim,
        #         self._horizon_of_observation_history + 1,
        #     ),
        #     dtype=np.float64,
        # )

        # updated_estimated_distribution_history[:, :, 0] = (
        #     estimated_distribution_history[:, :, 0].copy()
        # )

        # for k in range(1, self._horizon_of_observation_history+1):

        #     dual_function_history, control_history = _backward_path(
        #         estimated_distribution_history[..., :k+1],
        #         emission_history[..., :k+1],
        #         self._system.transition_matrix,
        #         self._terminal_dual_function,
        #     )

        #     updated_estimated_distribution_history[:, :, k] = _update_terminal_estimated_distribution(
        #         estimated_distribution_history[:, :, 0],
        #         dual_function_history[:, :, :, 0],
        #         control_history,
        #     )

        # The following code is the original implementation

        # dual_function_history = np.empty(
        #     (
        #         self._number_of_systems,
        #         self._discrete_state_dim,
        #         self._number_of_dual_function,
        #         self._horizon_of_observation_history + 1,
        #     ),
        #     dtype=np.float64,
        # )
        # dual_function_history[:, :, :, -1] = np.repeat(
        #     self._terminal_dual_function[np.newaxis, ...],
        #     self._number_of_systems,
        #     axis=0,
        # )

        # control_history = np.empty(
        #     (
        #         self._number_of_systems,
        #         self._number_of_dual_function,
        #         self._horizon_of_observation_history + 1,
        #     ),
        #     dtype=np.float64,
        # )

        # # Backward in time
        # # k = K, K-1, ..., 2, 1
        # for k in range(self._horizon_of_observation_history, 0, -1):
        #     dual_function = dual_function_history[
        #         :, :, :, k
        #     ]  # (number_of_systems, discrete_state_dim, number_of_dual_functions)
        #     emission = emission_history[
        #         :, :, k
        #     ]  # (number_of_systems, discrete_state_dim)

        #     past_estimated_distribution = estimated_distribution_history[
        #         :, :, k - 1
        #     ]  # (number_of_systems, discrete_state_dim)

        #     past_dual_function = _compute_past_dual_function(
        #         self._system.transition_matrix,
        #         dual_function,
        #     )

        #     # Compute the control
        #     control = _compute_control(
        #         past_dual_function,
        #         emission,
        #         past_estimated_distribution,
        #     )  # (number_of_systems, number_of_dual_functions)

        #     # Update the dual function
        #     dual_function_history[:, :, :, k - 1] = _backward_dual_function_step(
        #         past_dual_function,
        #         emission,
        #         control,
        #     )

        #     control_history[:, :, k] = control

        # estimator_history = np.empty(
        #     (
        #         self._number_of_systems,
        #         self._number_of_dual_function,
        #         self._horizon_of_observation_history + 1,
        #     ),
        #     dtype=np.float64,
        # )

        # updated_estimated_distribution_history = np.empty(
        #     (
        #         self._number_of_systems,
        #         self._discrete_state_dim,
        #         self._horizon_of_observation_history + 1,
        #     ),
        #     dtype=np.float64,
        # )

        # updated_estimated_distribution_history[:, :, 0] = (
        #     estimated_distribution_history[:, :, 0].copy()
        # )

        # # Compute the initial estimator
        # estimator_history[:, :, 0] = _compute_initial_estimator(
        #     updated_estimated_distribution_history[:, :, 0],
        #     dual_function_history[:, :, :, 0],
        # )  # (number_of_systems, number_of_dual_functions)

        # # Update the estimated distribution
        # # k = 1, 2, ..., K-1, K
        # for k in range(1, self._horizon_of_observation_history + 1):
        #     estimator_history[:, :, k] = _compute_estimator(
        #         estimator_history[:, :, k - 1],
        #         control_history[:, :, k],
        #     )

        #     updated_estimated_distribution_history[:, :, k] = (
        #         _update_estimated_distribution(
        #             dual_function_history[:, :, :, k],
        #             estimator_history[:, :, k],
        #         )
        #     )

        return control_history, updated_estimated_distribution_history


@njit(cache=True)  # type: ignore
def _backward_path(
    estimated_distribution_history: NDArray,
    emission_history: NDArray,
    transition_matrix: NDArray,
    terminal_dual_function: NDArray,
) -> Tuple[NDArray, NDArray]:
    number_of_systems, discrete_state_dim, horizon = (
        estimated_distribution_history.shape
    )
    horizon -= 1
    number_of_dual_functions = terminal_dual_function.shape[1]

    dual_function_history = np.empty(
        (
            number_of_systems,
            discrete_state_dim,
            number_of_dual_functions,
            horizon + 1,
        ),
        dtype=np.float64,
    )
    for i in range(number_of_systems):
        dual_function_history[i, :, :, -1] = terminal_dual_function.copy()

    control_history = np.full(
        (
            number_of_systems,
            number_of_dual_functions,
            horizon + 1,
        ),
        np.nan,
        dtype=np.float64,
    )

    for k in range(horizon, 0, -1):
        dual_function = dual_function_history[
            :, :, :, k
        ]  # (number_of_systems, discrete_state_dim, number_of_dual_functions)
        emission = emission_history[
            :, :, k
        ]  # (number_of_systems, discrete_state_dim)

        past_estimated_distribution = estimated_distribution_history[
            :, :, k - 1
        ]  # (number_of_systems, discrete_state_dim)

        past_dual_function = _compute_past_dual_function(
            transition_matrix,
            dual_function,
        )

        # Compute the control
        control = _compute_control(
            past_dual_function,
            emission,
            past_estimated_distribution,
        )  # (number_of_systems, number_of_dual_functions)

        # Update the dual function
        dual_function_history[:, :, :, k - 1] = _backward_dual_function_step(
            past_dual_function,
            emission,
            control,
        )

        control_history[:, :, k] = control

    return dual_function_history, control_history


@njit(cache=True)  # type: ignore
def _forward_path(
    dual_function_history: NDArray,
    control_history: NDArray,
    initial_estimated_distribution: NDArray,
) -> NDArray:
    (
        number_of_systems,
        discrete_state_dim,
        number_of_dual_functions,
        horizon,
    ) = dual_function_history.shape
    horizon -= 1

    estimator_history = np.empty(
        (
            number_of_systems,
            number_of_dual_functions,
            horizon + 1,
        ),
        dtype=np.float64,
    )

    updated_estimated_distribution_history = np.empty(
        (
            number_of_systems,
            discrete_state_dim,
            horizon + 1,
        ),
        dtype=np.float64,
    )
    updated_estimated_distribution_history[:, :, 0] = (
        initial_estimated_distribution.copy()
    )

    # Compute the initial estimator
    estimator_history[:, :, 0] = _compute_initial_estimator(
        updated_estimated_distribution_history[:, :, 0],
        dual_function_history[:, :, :, 0],
    )  # (number_of_systems, number_of_dual_functions)

    # Update the estimated distribution
    # k = 1, 2, ..., K-1, K
    for k in range(1, horizon + 1):

        estimator_history[:, :, k] = _compute_estimator(
            estimator_history[:, :, k - 1],
            control_history[:, :, k],
        )

        updated_estimated_distribution_history[:, :, k] = (
            _update_estimated_distribution(
                dual_function_history[:, :, :, k],
                estimator_history[:, :, k],
            )
        )
    return updated_estimated_distribution_history


@njit(cache=True)  # type: ignore
def _update_terminal_estimated_distribution(
    initial_estimated_distribution: NDArray,
    initial_dual_function: NDArray,
    control_history: NDArray,
) -> NDArray:

    number_of_systems, _, horizon = control_history.shape
    horizon -= 1
    updated_terminal_estimated_distribution: NDArray = (
        _compute_initial_estimator(
            initial_estimated_distribution,
            initial_dual_function,
        )
    )  # (number_of_systems, number_of_dual_functions)

    for k in range(1, horizon + 1):
        updated_terminal_estimated_distribution -= control_history[:, :, k]

    for i in range(number_of_systems):
        result = np.maximum(updated_terminal_estimated_distribution[i, :], 0)
        updated_terminal_estimated_distribution[i, :] = result / np.sum(result)

    return updated_terminal_estimated_distribution


@njit(cache=True)  # type: ignore
def _multi_path(
    estimated_distribution_history: NDArray,
    emission_history: NDArray,
    transition_matrix: NDArray,
    terminal_dual_function: NDArray,
) -> Tuple[NDArray, NDArray]:

    number_of_systems, discrete_state_dim, horizon = (
        estimated_distribution_history.shape
    )
    horizon -= 1

    updated_estimated_distribution_history = np.empty(
        (
            number_of_systems,
            discrete_state_dim,
            horizon + 1,
        ),
        dtype=np.float64,
    )

    updated_estimated_distribution_history[:, :, 0] = (
        estimated_distribution_history[:, :, 0].copy()
    )

    for k in range(1, horizon + 1):

        dual_function_history, control_history = _backward_path(
            estimated_distribution_history[..., : k + 1],
            emission_history[..., : k + 1],
            transition_matrix,
            terminal_dual_function,
        )

        updated_estimated_distribution_history[:, :, k] = (
            _update_terminal_estimated_distribution(
                estimated_distribution_history[:, :, 0],
                dual_function_history[:, :, :, 0],
                control_history,
            )
        )

    return control_history, updated_estimated_distribution_history


@njit(cache=True)  # type: ignore
def _terminal_path(
    estimated_distribution_history: NDArray,
    emission_history: NDArray,
    transition_matrix: NDArray,
    terminal_dual_function: NDArray,
) -> Tuple[NDArray, NDArray]:

    dual_function_history, control_history = _backward_path(
        estimated_distribution_history,
        emission_history,
        transition_matrix,
        terminal_dual_function,
    )

    terminal_estimated_distribution = _update_terminal_estimated_distribution(
        estimated_distribution_history[:, :, 0],
        dual_function_history[:, :, :, 0],
        control_history,
    )

    updated_estimated_distribution_history = (
        estimated_distribution_history.copy()
    )
    updated_estimated_distribution_history[:, :, -1] = (
        terminal_estimated_distribution.copy()
    )

    return control_history, updated_estimated_distribution_history


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


@njit(cache=True)  # type: ignore
def _compute_control(
    past_dual_function: NDArray[np.float64],
    emission: NDArray[np.float64],
    estimated_distribution: NDArray[np.float64],
) -> NDArray[np.float64]:
    number_of_systems, _, number_of_dual_functions = past_dual_function.shape
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
            # The following implementation is equivalent to the one below (but faster?)
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


@njit(cache=True)  # type: ignore
def _backward_dual_function_step(
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
    return updated_past_dual_function


@njit(cache=True)  # type: ignore
def _compute_initial_estimator(
    initial_distribution: NDArray[np.float64],
    dual_function: NDArray[np.float64],
) -> NDArray[np.float64]:
    number_of_systems, _, number_of_dual_functions = dual_function.shape
    initial_estimator = np.empty((number_of_systems, number_of_dual_functions))
    for i in range(number_of_systems):
        for d in range(number_of_dual_functions):
            initial_estimator[i, d] = np.sum(
                initial_distribution[i, :] * dual_function[i, :, d]
            )
    return initial_estimator


@njit(cache=True)  # type: ignore
def _compute_estimator(
    past_estimator: NDArray[np.float64],
    control: NDArray[np.float64],
) -> NDArray[np.float64]:
    return past_estimator - control


@njit(cache=True)  # type: ignore
def _update_estimated_distribution(
    dual_function: NDArray[np.float64],
    estimator: NDArray[np.float64],
) -> NDArray[np.float64]:
    number_of_systems, discrete_state_dim, _ = dual_function.shape
    updated_estimated_distribution = np.empty(
        (number_of_systems, discrete_state_dim)
    )
    for i in range(number_of_systems):
        result, _, _, _ = np.linalg.lstsq(
            dual_function[i, :, :].T,
            estimator[i, :],
        )  # (discrete_state_dim,)
        result = np.maximum(result, 0)
        updated_estimated_distribution[i, :] = result / np.sum(result)
    return updated_estimated_distribution


class DualHmmFilterCallback(Callback):
    def __init__(
        self,
        step_skip: int,
        filter: DualHmmFilter,
    ) -> None:
        assert issubclass(type(filter), DualHmmFilter), (
            f"filter must be an instance of DualHmmFilter or its subclasses. "
            f"filter given is an instance of {type(filter)}."
        )
        self._filter = filter
        super().__init__(step_skip)

    def _record(self, time: float) -> None:
        super()._record(time)
        self._callback_params["estimated_distribution_history"].append(
            self._filter.estimated_distribution_history.copy()
        )
        self._callback_params["control_history"].append(
            self._filter.control_history.copy()
        )
