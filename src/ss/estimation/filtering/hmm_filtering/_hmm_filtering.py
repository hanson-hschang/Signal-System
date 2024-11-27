from typing import Optional

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ss.estimation.filtering import Filter
from ss.system.finite_state.markov import (
    HiddenMarkovModel,
    one_hot_decoding,
    one_hot_encoding,
)


class HiddenMarkovModelFilter(Filter):
    def __init__(
        self,
        system: HiddenMarkovModel,
        initial_distribution: Optional[ArrayLike] = None,
    ) -> None:
        assert issubclass(type(system), HiddenMarkovModel)
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
            f"initial_distribution should have the same length as state_dim {self._state_dim}."
            f"The initial_distribution given has shape {initial_distribution.shape}."
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
        observation_value: NDArray[np.int64] = one_hot_decoding(observation)
        estimated_state[...] = (
            estimated_state
            * emission_probability_matrix[:, observation_value].T
        ) @ transition_probability_matrix
        for i in range(number_of_systems):
            estimated_state[i, :] /= np.sum(estimated_state[i, :])
        return estimated_state
