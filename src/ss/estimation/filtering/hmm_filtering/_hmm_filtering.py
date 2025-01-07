from typing import Callable, Optional

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ss.estimation import EstimatorCallback
from ss.estimation.filtering import Filter
from ss.system.markov import HiddenMarkovModel, one_hot_decoding
from ss.utility.descriptor import MultiSystemNDArrayReadOnlyDescriptor


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
