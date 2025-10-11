import numpy as np
from numba import njit, prange
from numpy.typing import ArrayLike, NDArray

from ss.estimation import EstimatorCallback
from ss.estimation.filtering import DualFilter
from ss.system.markov import HiddenMarkovModel, get_estimation_model
from ss.utility.descriptor import BatchNDArrayReadOnlyDescriptor
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class DualHmmFilter(DualFilter):
    def __init__(
        self,
        system: HiddenMarkovModel,
        history_horizon: int,
        initial_distribution: ArrayLike | None = None,
        estimation_matrix: ArrayLike | None = None,
        batch_size: int | None = None,
    ) -> None:
        self._system = system
        self._estimation_matrix = (
            np.identity(self._system.discrete_state_dim)
            if estimation_matrix is None
            else np.array(estimation_matrix)
        )

        super().__init__(
            state_dim=self._system.discrete_state_dim,
            observation_dim=self._system.observation_dim,
            history_horizon=history_horizon,
            initial_distribution=initial_distribution,
            estimation_model=get_estimation_model(
                matrix=self._estimation_matrix
            ),
            batch_size=system.batch_size if batch_size is None else batch_size,
        )
        self._discrete_observation_dim = self._system.discrete_observation_dim

        self._terminal_dual_function = np.identity(
            self._system.discrete_state_dim
        )
        self._number_of_dual_function = self._terminal_dual_function.shape[1]

        self._emission_history = np.full(
            (
                self._batch_size,
                self._state_dim,
                self._history_horizon,
            ),
            np.nan,
            dtype=np.float64,
        )
        self._state_history_horizon = self._history_horizon + 1
        self._estimated_state_history = np.full(
            (
                self._batch_size,
                self._state_dim,
                self._state_history_horizon,
            ),
            np.nan,
            dtype=np.float64,
        )

        self._control_history = np.full(
            (
                self._batch_size,
                self._number_of_dual_function,
                self._history_horizon,
            ),
            np.nan,
            dtype=np.float64,
        )

        self.reset()

    emission_history = BatchNDArrayReadOnlyDescriptor(
        "_batch_size",
        "_state_dim",
        "_history_horizon",
    )
    estimated_state_history = BatchNDArrayReadOnlyDescriptor(
        "_batch_size",
        "_state_dim",
        "_state_history_horizon",
    )
    control_history = BatchNDArrayReadOnlyDescriptor(
        "_batch_size",
        "_number_of_dual_function",
        "_history_horizon",
    )

    def reset(self) -> None:
        super().reset()
        self._emission_history[:, :, :] = np.nan
        self._estimated_state_history[:, :, :] = np.nan
        self._estimated_state_history[:, :, 0] = self._initial_distribution[
            np.newaxis, :
        ]
        self._control_history[:, :, :] = np.nan
        self._estimated_state[:, :] = self._initial_distribution[np.newaxis, :]
        self.estimate()

    def _compute_estimated_state_process(self) -> NDArray[np.float64]:
        # Update the emission history based on the observation history
        self._update_emission(
            self._system.emission_matrix,
            self._observation_history[:, 0, 0].astype(np.int64),
            self._emission_history,
        )

        # Compute the dual process (include control)
        dual_function_history = self._compute_backward_path(
            self._system.transition_matrix,
            self._terminal_dual_function,
            self._emission_history[:, :, : self._current_history_horizon],
            self._estimated_state_history[
                :, :, : self._current_history_horizon
            ],
            self._control_history,
        )

        # Compute the estimated state distribution
        estimated_state_distribution: NDArray[np.float64] = (
            self._compute_estimated_state_distribution(
                self._estimated_state_history[
                    :, :, self._current_history_horizon - 1
                ],
                dual_function_history[:, :, :, -1],
                self._control_history[:, :, : self._current_history_horizon],
            )
        )

        # Update the estimated state distribution history
        self._update_estimated_state_distribution(
            estimated_state_distribution,
            self._estimated_state_history,
        )

        return estimated_state_distribution

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _update_emission(
        emission_matrix: NDArray[np.float64],
        observation: NDArray[np.int64],
        emission_history: NDArray[np.float64],
    ) -> None:
        # The updated emission history will have the following structure:
        # emission_history[:, :, 0] : the most recent emission
        # emission_history[:, :, 1] : the second most recent emission
        # ...
        # emission_history[:, :, -1] : the oldest emission

        batch_size, discrete_state_dim, horizon = emission_history.shape

        # Replace the oldest emission with the most recent one.
        emission_column = emission_matrix[
            :, observation
        ].T  # (batch_size, discrete_state_dim)
        emission_history[:, :, -1] = 2 * emission_column - 1

        if horizon > 1:
            # Shift the emission history to the right by one.
            for i in prange(batch_size):
                for d in prange(discrete_state_dim):
                    emission_history[i, d, :] = np.roll(
                        emission_history[i, d, :], 1
                    )
        # If horizon == 1, no need to shift the emission history.

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _update_estimated_state_distribution(
        estimated_state_distribution: NDArray[np.float64],
        estimated_state_distribution_history: NDArray[np.float64],
    ) -> None:
        batch_size, state_dim, horizon = (
            estimated_state_distribution_history.shape
        )

        # Replace the oldest estimated distribution with the most recent one.
        estimated_state_distribution_history[:, :, -1] = (
            estimated_state_distribution
        )

        if horizon > 1:
            # Shift the estimated distribution history to the right by one.
            for i in prange(batch_size):
                for d in prange(state_dim):
                    estimated_state_distribution_history[i, d, :] = np.roll(
                        estimated_state_distribution_history[i, d, :], 1
                    )
        # If horizon == 1, no need to shift the estimated distribution history.

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _compute_estimated_state_distribution(
        initial_estimated_state_distribution: NDArray[np.float64],
        initial_dual_function: NDArray[np.float64],
        control_history: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        batch_size, state_dim = initial_estimated_state_distribution.shape
        estimated_state_distribution = np.empty(
            (batch_size, state_dim), dtype=np.float64
        )
        for i in prange(batch_size):
            _initial_estimated_state_distribution = np.ascontiguousarray(
                initial_estimated_state_distribution[i, :]
            )
            _initial_dual_function = np.ascontiguousarray(
                initial_dual_function[i, :, :]
            )
            estimated_state_distribution[i, :] = (
                _initial_estimated_state_distribution @ _initial_dual_function
            )
        estimated_state_distribution[:, :] = (
            estimated_state_distribution - np.sum(control_history, axis=2)
        )
        return estimated_state_distribution

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _compute_backward_path(
        transition_matrix: NDArray[np.float64],
        terminal_dual_function: NDArray[np.float64],
        emission_history: NDArray[np.float64],
        estimated_state_distribution_history: NDArray[np.float64],
        control_history: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        state_dim, number_of_dual_function = terminal_dual_function.shape
        batch_size, _, horizon = emission_history.shape
        dual_function_history = np.empty(
            (batch_size, state_dim, number_of_dual_function, horizon + 1),
            dtype=np.float64,
        )
        dual_function_history[:, :, :, 0] = terminal_dual_function[
            np.newaxis, :, :
        ]

        for k in range(horizon):
            dual_function = _compute_past_dual_function(
                transition_matrix,
                dual_function_history[:, :, :, k],
            )
            emission = emission_history[:, :, k]

            control = _compute_control(
                dual_function,
                emission,
                estimated_state_distribution_history[:, :, k],
            )

            dual_function_history[:, :, :, k + 1] = (
                _backward_dual_function_step(
                    dual_function,
                    emission,
                    control,
                )
            )
            control_history[:, :, k] = control

        return dual_function_history


@njit(cache=True)  # type: ignore
def _compute_past_dual_function(
    transition_matrix: NDArray[np.float64],
    dual_function: NDArray[np.float64],
) -> NDArray[np.float64]:
    batch_size = dual_function.shape[0]
    past_dual_function = np.empty_like(dual_function)
    for i in prange(batch_size):
        _dual_function = np.ascontiguousarray(dual_function[i, :, :])
        past_dual_function[i, :, :] = transition_matrix @ _dual_function
    return past_dual_function


@njit(cache=True)  # type: ignore
def _compute_control(
    past_dual_function: NDArray[np.float64],
    emission: NDArray[np.float64],
    estimated_distribution: NDArray[np.float64],
) -> NDArray[np.float64]:
    batch_size, _, number_of_dual_functions = past_dual_function.shape
    control = np.empty((batch_size, number_of_dual_functions))
    for i in prange(batch_size):
        expected_emission = np.sum(
            estimated_distribution[i, :] * emission[i, :]
        )
        denominator = 1 - (expected_emission**2)

        if denominator == 0:
            control[i, :] = 0
            continue

        for d in prange(number_of_dual_functions):
            # The following implementation is equivalent
            # to the one below (but faster?)
            control[i, d] = (
                -(
                    np.sum(
                        estimated_distribution[i, :]
                        * (
                            past_dual_function[i, :, d]
                            * (emission[i, :] - expected_emission)
                        )
                    )
                )
                / denominator
            )
            # control[i, d] = (
            #     (
            #         np.sum(
            #             estimated_distribution[i, :]
            #             * past_dual_function[i, :, d]
            #         )
            #         * expected_emission
            #     )
            #     - np.sum(
            #         estimated_distribution[i, :]
            #         * (past_dual_function[i, :, d] * emission[i, :])
            #     )
            # ) / denominator
    return control


@njit(cache=True)  # type: ignore
def _backward_dual_function_step(
    past_dual_function: NDArray[np.float64],
    emission: NDArray[np.float64],
    control: NDArray[np.float64],
) -> NDArray[np.float64]:
    batch_size = past_dual_function.shape[0]
    updated_past_dual_function = np.empty_like(past_dual_function)
    for i in prange(batch_size):
        updated_past_dual_function[i, :, :] = past_dual_function[
            i, :, :
        ] + np.outer(emission[i, :], control[i, :])
    return updated_past_dual_function


class DualHmmFilterCallback(EstimatorCallback[DualHmmFilter]):
    def __init__(
        self,
        step_skip: int,
        filter: DualHmmFilter,
    ) -> None:
        assert issubclass(type(filter), DualHmmFilter), (
            f"filter must be an instance of DualHmmFilter or its subclasses. "
            f"filter given is an instance of {type(filter)}."
        )
        super().__init__(step_skip, filter)

    def _record(self, time: float) -> None:
        super()._record(time)
        self._callback_params["control_history"].append(
            self._estimator.control_history.copy()
        )
