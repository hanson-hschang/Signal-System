from typing import Optional, Tuple

import numpy as np
from numba import njit, prange
from numpy.typing import ArrayLike, NDArray

# from ss.utility.callback import Callback
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
        horizon_of_observation_history: int,
        initial_distribution: Optional[ArrayLike] = None,
        estimation_matrix: Optional[ArrayLike] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        self._system = system
        self._estimation_matrix = (
            np.identity(self._system.discrete_state_dim)
            if estimation_matrix is None
            else np.array(estimation_matrix)
        )
        # self._current_horizon_of_observation_history = 0
        # self._horizon_of_observation_history = horizon_of_observation_history
        super().__init__(
            state_dim=self._system.discrete_state_dim,
            observation_dim=self._system.observation_dim,
            horizon_of_observation_history=horizon_of_observation_history,
            initial_distribution=initial_distribution,
            estimation_model=get_estimation_model(
                matrix=self._estimation_matrix
            ),
            batch_size=system.batch_size if batch_size is None else batch_size,
        )
        # self._state_dim = self._system.discrete_state_dim
        # self._observation_dim = self._system.observation_dim
        # assert self._observation_dim == 1, (
        #     f"observation_dim must be 1. "
        #     f"observation_dim given is {self._observation_dim}."
        # )
        self._discrete_observation_dim = self._system.discrete_observation_dim

        # self._batch_size = system.batch_size if batch_size is None else batch_size

        # self._initial_distribution = np.repeat(
        #     (
        #         np.full(self._state_dim, 1 / self._state_dim)
        #         if initial_distribution is None
        #         else np.array(initial_distribution)
        #     )[np.newaxis, :],
        #     self._batch_size,
        #     axis=0,
        # )

        self._terminal_dual_function = np.identity(
            self._system.discrete_state_dim
        )
        # (

        #     if estimation_model is None
        #     else np.array(estimation_model)
        # )
        self._number_of_dual_function = self._terminal_dual_function.shape[1]

        # self._observation_history = np.full(
        #     (
        #         self._batch_size,
        #         self._observation_dim,
        #         self._horizon_of_observation_history,
        #     ),
        #     np.nan,
        #     dtype=np.float64,
        # )
        self._emission_history = np.full(
            (
                self._batch_size,
                self._state_dim,
                self._horizon_of_observation_history,
            ),
            np.nan,
            dtype=np.float64,
        )
        # self._likelihood_history = np.full(
        #     (
        #         self._batch_size,
        #         self._state_dim,
        #         self._horizon_of_observation_history,
        #     ),
        #     np.nan,
        #     dtype=np.float64,
        # )
        # self._likelihood_history[:, :, -1] = self._initial_distribution.copy()

        self._horizon_of_state_history = (
            self._horizon_of_observation_history + 1
        )
        self._estimated_state_distribution_history = np.full(
            (
                self._batch_size,
                self._state_dim,
                self._horizon_of_state_history,
            ),
            np.nan,
            dtype=np.float64,
        )
        # self._estimated_state_distribution_history[:, :, 0] = (
        #     self._initial_distribution[np.newaxis, :]
        # )

        self._control_history = np.full(
            (
                self._batch_size,
                self._number_of_dual_function,
                self._horizon_of_observation_history,
            ),
            np.nan,
            dtype=np.float64,
        )

        self.reset()

    # observation_history = BatchNDArrayReadOnlyDescriptor(
    #     "_batch_size",
    #     "_observation_dim",
    #     "_horizon_of_observation_history",
    # )
    emission_history = BatchNDArrayReadOnlyDescriptor(
        "_batch_size",
        "_state_dim",
        "_horizon_of_observation_history",
    )
    # likelihood_history = BatchNDArrayReadOnlyDescriptor(
    #     "_batch_size",
    #     "_state_dim",
    #     "_horizon_of_observation_history",
    # )
    estimated_state_distribution_history = BatchNDArrayReadOnlyDescriptor(
        "_batch_size",
        "_state_dim",
        "_horizon_of_state_history",
    )
    control_history = BatchNDArrayReadOnlyDescriptor(
        "_batch_size",
        "_number_of_dual_function",
        "_horizon_of_observation_history",
    )

    def reset(self) -> None:
        super().reset()
        self._emission_history[:, :, :] = np.nan
        self._estimated_state_distribution_history[:, :, :] = np.nan
        self._estimated_state_distribution_history[:, :, 0] = (
            self._initial_distribution[np.newaxis, :]
        )
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
            self._emission_history[
                :, :, : self._current_horizon_of_observation_history
            ],
            self._estimated_state_distribution_history[
                :, :, : self._current_horizon_of_observation_history
            ],
            self._control_history,
        )

        # Compute the estimated state distribution
        estimated_state_distribution: NDArray[np.float64] = (
            self._compute_estimated_state_distribution(
                self._estimated_state_distribution_history[
                    :, :, self._current_horizon_of_observation_history - 1
                ],
                dual_function_history[:, :, :, -1],
                self._control_history[
                    :, :, : self._current_horizon_of_observation_history
                ],
            )
        )

        # Update the estimated state distribution history
        self._update_estimated_state_distribution(
            estimated_state_distribution,
            self._estimated_state_distribution_history,
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

        # for i in prange(batch_size):
        #     _terminal_dual_function = np.ascontiguousarray(terminal_dual_function[i, :, :])
        #     _emission_history = np.ascontiguousarray(emission_history[i, :, :current_horizon_of_observation_history+1])
        #     _estimated_state_distribution_history = np.ascontiguousarray(estimated_state_distribution_history[i, :, :current_horizon_of_observation_history+1])
        #     _control_history = np.ascontiguousarray(control_history[i, :, :current_horizon_of_observation_history+1])
        #     initial_dual_function[i, :, :] = _terminal_path(
        #         _terminal_dual_function,
        #         _emission_history,
        #         _estimated_state_distribution_history,
        #         _control_history,
        #         current_horizon_of_observation_history,
        #     )
        return dual_function_history

    # def update(self, observation: ArrayLike) -> None:
    #     """
    #     Update the observation history with the given observation.

    #     Parameters
    #     ----------
    #     observation : ArrayLike
    #         shape = (batch_size, observation_dim)
    #         The observation to be updated.
    #     """
    #     observation = np.array(observation, dtype=np.int64)
    #     if observation.ndim == 1:
    #         observation = observation[np.newaxis, :]
    #     assert observation.shape == (
    #         self._batch_size,
    #         self._observation_dim,
    #     ), (
    #         f"observation must be in the shape of {(self._batch_size, self._observation_dim) = }. "
    #         f"observation given has the shape of {observation.shape}."
    #     )

    #     self._update_observation(
    #         observation=observation,
    #         emission_matrix=self._system.emission_matrix,
    #         initial_distribution=self._initial_distribution,
    #         observation_history=self._observation_history,
    #         emission_history=self._emission_history,
    #         likelihood_history=self._likelihood_history,
    #         estimated_distribution_history=self._estimated_distribution_history,
    #     )
    #     self._current_horizon_of_observation_history = min(
    #         self._current_horizon_of_observation_history + 1,
    #         self._horizon_of_observation_history - 1,
    #     )

    # @staticmethod
    # @njit(cache=True)  # type: ignore
    # def _update_observation(
    #     observation: NDArray[np.int64],
    #     emission_matrix: NDArray[np.float64],
    #     initial_distribution: NDArray[np.float64],
    #     observation_history: NDArray[np.float64],
    #     emission_history: NDArray[np.float64],
    #     likelihood_history: NDArray[np.float64],
    #     estimated_distribution_history: NDArray[np.float64],
    # ) -> None:
    #     batch_size, _, _ = observation_history.shape
    #     discrete_state_dim, _ = emission_matrix.shape

    #     # Move the observation, emission, and estimated distribution history one step into the past
    #     for i in prange(batch_size):
    #         # numba v0.61.0 does not support np.roll with axis argument
    #         observation_history[i, 0, :] = np.roll(
    #             observation_history[i, 0, :], shift=-1
    #         )
    #         for d in prange(discrete_state_dim):
    #             emission_history[i, d, :] = np.roll(
    #                 emission_history[i, d, :], shift=-1
    #             )
    #             likelihood_history[i, d, :] = np.roll(
    #                 likelihood_history[i, d, :], shift=-1
    #             )
    #             estimated_distribution_history[i, d, :] = np.roll(
    #                 estimated_distribution_history[i, d, :], shift=-1
    #             )
    #     # Update the most recent history based on the new observation
    #     emission_column = emission_matrix[
    #         :, observation[:, 0]
    #     ].T  # (batch_size, discrete_state_dim)
    #     emission_history[:, :, -1] = 2 * emission_column - 1
    #     for i in prange(batch_size):
    #         # numba v0.61.0 does not support np.sum with axis argument
    #         likelihood_history[i, :, -1] = emission_column[i, :] / np.sum(
    #             emission_column[i, :]
    #         )
    #         if likelihood_history[i, 0, 0] != np.nan:
    #             likelihood_history[i, :, 0] = initial_distribution[i, :].copy()
    #         if emission_history[i, 0, 0] != np.nan:
    #             emission_history[i, :, 0] = np.nan
    #         if observation_history[i, 0, 0] != np.nan:
    #             observation_history[i, 0, 0] = np.nan
    #     estimated_distribution_history[:, :, -1] = likelihood_history[
    #         :, :, -1
    #     ].copy()
    #     observation_history[:, :, -1] = observation.copy()

    # def estimate(
    #     self, iterations: int = 1, show_progress: bool = False
    # ) -> Tuple[NDArray, NDArray]:
    #     """
    #     Estimate the distribution of the system.

    #     Parameters
    #     ----------
    #     iterations : int
    #         The number of iterations to run the estimation.

    #     Returns
    #     -------
    #     estimated_distribution : NDArray
    #         The estimated distribution of the system.
    #     """
    #     number_of_dual_functions = self._terminal_dual_function.shape[1]
    #     control_history = np.empty(
    #         (
    #             iterations,
    #             self._batch_size,
    #             number_of_dual_functions,
    #             self._current_horizon_of_observation_history,
    #         ),
    #     )
    #     estimated_distribution_history = np.empty(
    #         (
    #             iterations + 1,
    #             self._batch_size,
    #             self._state_dim,
    #             self._current_horizon_of_observation_history + 1,
    #         ),
    #     )
    #     # estimated_distribution_history[0, ...] = self._likelihood_history[
    #     #     ..., -1 - self._current_horizon_of_observation_history :
    #     # ].copy()

    #     for i in logger.progress_bar(
    #         range(iterations), show_progress=show_progress
    #     ):
    #         (
    #             control_history[i, ...],
    #             estimated_distribution_history[i + 1, ...],
    #         ) = self._estimate(
    #             estimated_distribution_history[i, ...],
    #             self._emission_history[..., -1 - self._current_horizon_of_observation_history :],
    #         )

    #     self._control_history[..., -1 - self._current_horizon_of_observation_history :] = control_history[
    #         -1, ...
    #     ].copy()

    #     self._estimated_state_distribution_history[..., -1 - self._current_horizon_of_observation_history :] = (
    #         estimated_distribution_history[-1, ...].copy()
    #     )

    #     if self._current_horizon_of_observation_history == self._horizon_of_observation_history - 1:
    #         self._initial_distribution = self._estimated_state_distribution_history[
    #             ..., -self._current_horizon_of_observation_history
    #         ].copy()

    #     return control_history, estimated_distribution_history

    # def _estimate(
    #     self,
    #     estimated_distribution_history: NDArray,
    #     emission_history: NDArray,
    # ) -> Tuple[NDArray, NDArray]:

    # control_history, updated_estimated_distribution_history = _terminal_path(
    #     estimated_distribution_history,
    #     emission_history,
    #     self._system.transition_matrix,
    #     self._terminal_dual_function,
    # )

    # control_history, updated_estimated_distribution_history = _multi_path(
    #     estimated_distribution_history,
    #     emission_history,
    #     self._system.transition_matrix,
    #     self._terminal_dual_function,
    # )

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
    #         self._batch_size,
    #         self._state_dim,
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
    #         self._batch_size,
    #         self._state_dim,
    #         self._number_of_dual_function,
    #         self._horizon_of_observation_history + 1,
    #     ),
    #     dtype=np.float64,
    # )
    # dual_function_history[:, :, :, -1] = np.repeat(
    #     self._terminal_dual_function[np.newaxis, ...],
    #     self._batch_size,
    #     axis=0,
    # )

    # control_history = np.empty(
    #     (
    #         self._batch_size,
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
    #     ]  # (batch_size, discrete_state_dim, number_of_dual_functions)
    #     emission = emission_history[
    #         :, :, k
    #     ]  # (batch_size, discrete_state_dim)

    #     past_estimated_distribution = estimated_distribution_history[
    #         :, :, k - 1
    #     ]  # (batch_size, discrete_state_dim)

    #     past_dual_function = _compute_past_dual_function(
    #         self._system.transition_matrix,
    #         dual_function,
    #     )

    #     # Compute the control
    #     control = _compute_control(
    #         past_dual_function,
    #         emission,
    #         past_estimated_distribution,
    #     )  # (batch_size, number_of_dual_functions)

    #     # Update the dual function
    #     dual_function_history[:, :, :, k - 1] = _backward_dual_function_step(
    #         past_dual_function,
    #         emission,
    #         control,
    #     )

    #     control_history[:, :, k] = control

    # estimator_history = np.empty(
    #     (
    #         self._batch_size,
    #         self._number_of_dual_function,
    #         self._horizon_of_observation_history + 1,
    #     ),
    #     dtype=np.float64,
    # )

    # updated_estimated_distribution_history = np.empty(
    #     (
    #         self._batch_size,
    #         self._state_dim,
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
    # )  # (batch_size, number_of_dual_functions)

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

    # return control_history, updated_estimated_distribution_history


# @njit(cache=True)  # type: ignore
# def _backward_path(
#     estimated_distribution_history: NDArray,
#     emission_history: NDArray,
#     transition_matrix: NDArray,
#     terminal_dual_function: NDArray,
# ) -> Tuple[NDArray, NDArray]:
#     batch_size, discrete_state_dim, horizon = (
#         estimated_distribution_history.shape
#     )
#     horizon -= 1
#     number_of_dual_functions = terminal_dual_function.shape[1]

#     dual_function_history = np.empty(
#         (
#             batch_size,
#             discrete_state_dim,
#             number_of_dual_functions,
#             horizon + 1,
#         ),
#         dtype=np.float64,
#     )
#     for i in prange(batch_size):
#         dual_function_history[i, :, :, -1] = terminal_dual_function.copy()

#     control_history = np.full(
#         (
#             batch_size,
#             number_of_dual_functions,
#             horizon + 1,
#         ),
#         np.nan,
#         dtype=np.float64,
#     )

#     for k in range(horizon, 0, -1):
#         dual_function = dual_function_history[
#             :, :, :, k
#         ]  # (batch_size, discrete_state_dim, number_of_dual_functions)
#         emission = emission_history[
#             :, :, k
#         ]  # (batch_size, discrete_state_dim)

#         past_estimated_distribution = estimated_distribution_history[
#             :, :, k - 1
#         ]  # (batch_size, discrete_state_dim)

#         past_dual_function = _compute_past_dual_function(
#             transition_matrix,
#             dual_function,
#         )

#         # Compute the control
#         control = _compute_control(
#             past_dual_function,
#             emission,
#             past_estimated_distribution,
#         )  # (batch_size, number_of_dual_functions)

#         # Update the dual function
#         dual_function_history[:, :, :, k - 1] = _backward_dual_function_step(
#             past_dual_function,
#             emission,
#             control,
#         )

#         control_history[:, :, k] = control

#     return dual_function_history, control_history


# @njit(cache=True)  # type: ignore
# def _forward_path(
#     dual_function_history: NDArray,
#     control_history: NDArray,
#     initial_estimated_distribution: NDArray,
# ) -> NDArray:
#     (
#         batch_size,
#         discrete_state_dim,
#         number_of_dual_functions,
#         horizon,
#     ) = dual_function_history.shape
#     horizon -= 1

#     estimator_history = np.empty(
#         (
#             batch_size,
#             number_of_dual_functions,
#             horizon + 1,
#         ),
#         dtype=np.float64,
#     )

#     updated_estimated_distribution_history = np.empty(
#         (
#             batch_size,
#             discrete_state_dim,
#             horizon + 1,
#         ),
#         dtype=np.float64,
#     )
#     updated_estimated_distribution_history[:, :, 0] = (
#         initial_estimated_distribution.copy()
#     )

#     # Compute the initial estimator
#     estimator_history[:, :, 0] = _compute_initial_estimator(
#         updated_estimated_distribution_history[:, :, 0],
#         dual_function_history[:, :, :, 0],
#     )  # (batch_size, number_of_dual_functions)

#     # Update the estimated distribution
#     # k = 1, 2, ..., K-1, K
#     for k in range(1, horizon + 1):

#         estimator_history[:, :, k] = _compute_estimator(
#             estimator_history[:, :, k - 1],
#             control_history[:, :, k],
#         )

#         updated_estimated_distribution_history[:, :, k] = (
#             _update_estimated_distribution(
#                 dual_function_history[:, :, :, k],
#                 estimator_history[:, :, k],
#             )
#         )
#     return updated_estimated_distribution_history


# @njit(cache=True)  # type: ignore
# def _update_terminal_estimated_distribution(
#     initial_estimated_distribution: NDArray,
#     initial_dual_function: NDArray,
#     control_history: NDArray,
# ) -> NDArray:

#     batch_size, number_of_dual_functions, horizon = control_history.shape
#     horizon -= 1
#     updated_terminal_estimated_distribution: NDArray = (
#         _compute_initial_estimator(
#             initial_estimated_distribution,
#             initial_dual_function,
#         )
#     )  # (batch_size, number_of_dual_functions)

#     for k in range(1, horizon + 1):
#         updated_terminal_estimated_distribution -= control_history[:, :, k]

#     for i in prange(batch_size):
#         positive_mask = updated_terminal_estimated_distribution[i, :] > 0
#         # If all values are negative, set the distribution to be uniform
#         if np.all(np.logical_not(positive_mask)):
#             updated_terminal_estimated_distribution[i, :] = (
#                 1 / number_of_dual_functions
#             )
#             continue
#         # min_nonnegative_value = np.min(
#         #     updated_terminal_estimated_distribution[i, positive_mask]
#         # )
#         min_nonnegative_value = 0.0
#         result = np.maximum(
#             updated_terminal_estimated_distribution[i, :],
#             min_nonnegative_value,
#         )
#         updated_terminal_estimated_distribution[i, :] = result / np.sum(result)

#     return updated_terminal_estimated_distribution


# @njit(cache=True)  # type: ignore
# def _multi_path(
#     estimated_distribution_history: NDArray,
#     emission_history: NDArray,
#     transition_matrix: NDArray,
#     terminal_dual_function: NDArray,
# ) -> Tuple[NDArray, NDArray]:

#     batch_size, discrete_state_dim, horizon = (
#         estimated_distribution_history.shape
#     )
#     horizon -= 1

#     updated_estimated_distribution_history = np.empty(
#         (
#             batch_size,
#             discrete_state_dim,
#             horizon + 1,
#         ),
#         dtype=np.float64,
#     )

#     updated_estimated_distribution_history[:, :, 0] = (
#         estimated_distribution_history[:, :, 0].copy()
#     )

#     for k in range(1, horizon + 1):
#         _estimated_distribution_history = estimated_distribution_history[
#             :, :, : k + 1
#         ].copy()
#         _estimated_distribution_history[:, :, :k] = (
#             updated_estimated_distribution_history[:, :, :k].copy()
#         )

#         dual_function_history, control_history = _backward_path(
#             _estimated_distribution_history,
#             emission_history[..., : k + 1],
#             transition_matrix,
#             terminal_dual_function,
#         )

#         updated_estimated_distribution_history[:, :, k] = (
#             _update_terminal_estimated_distribution(
#                 estimated_distribution_history[:, :, 0],
#                 dual_function_history[:, :, :, 0],
#                 control_history,
#             )
#         )

#     return control_history, updated_estimated_distribution_history


# @njit(cache=True)  # type: ignore
# def _terminal_path(
#     estimated_distribution_history: NDArray,
#     emission_history: NDArray,
#     transition_matrix: NDArray,
#     terminal_dual_function: NDArray,
# ) -> Tuple[NDArray, NDArray]:

#     dual_function_history, control_history = _backward_path(
#         estimated_distribution_history,
#         emission_history,
#         transition_matrix,
#         terminal_dual_function,
#     )

#     terminal_estimated_distribution = _update_terminal_estimated_distribution(
#         estimated_distribution_history[:, :, 0],
#         dual_function_history[:, :, :, 0],
#         control_history,
#     )

#     updated_estimated_distribution_history = (
#         estimated_distribution_history.copy()
#     )
#     updated_estimated_distribution_history[:, :, -1] = (
#         terminal_estimated_distribution.copy()
#     )

#     return control_history, updated_estimated_distribution_history


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
    batch_size = past_dual_function.shape[0]
    updated_past_dual_function = np.empty_like(past_dual_function)
    for i in prange(batch_size):
        updated_past_dual_function[i, :, :] = past_dual_function[
            i, :, :
        ] + np.outer(emission[i, :], control[i, :])
    return updated_past_dual_function


# @njit(cache=True)  # type: ignore
# def _compute_initial_estimator(
#     initial_distribution: NDArray[np.float64],
#     dual_function: NDArray[np.float64],
# ) -> NDArray[np.float64]:
#     batch_size, _, number_of_dual_functions = dual_function.shape
#     initial_estimator = np.empty((batch_size, number_of_dual_functions))
#     for i in prange(batch_size):
#         for d in prange(number_of_dual_functions):
#             initial_estimator[i, d] = np.sum(
#                 initial_distribution[i, :] * dual_function[i, :, d]
#             )
#     return initial_estimator


# @njit(cache=True)  # type: ignore
# def _compute_estimator(
#     past_estimator: NDArray[np.float64],
#     control: NDArray[np.float64],
# ) -> NDArray[np.float64]:
#     return past_estimator - control


# @njit(cache=True)  # type: ignore
# def _update_estimated_distribution(
#     dual_function: NDArray[np.float64],
#     estimator: NDArray[np.float64],
# ) -> NDArray[np.float64]:
#     batch_size, discrete_state_dim, _ = dual_function.shape
#     updated_estimated_distribution = np.empty((batch_size, discrete_state_dim))
#     for i in prange(batch_size):
#         result, _, _, _ = np.linalg.lstsq(
#             dual_function[i, :, :].T,
#             estimator[i, :],
#         )  # (discrete_state_dim,)
#         result = np.maximum(result, 0)
#         updated_estimated_distribution[i, :] = result / np.sum(result)
#     return updated_estimated_distribution


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
        # self._filter = filter
        super().__init__(step_skip, filter)

    def _record(self, time: float) -> None:
        super()._record(time)
        self._callback_params["control_history"].append(
            self._estimator.control_history.copy()
        )
