import numpy as np
from numpy.typing import NDArray

from ss.estimation.filtering import Filter
from ss.system.finite_state.markov import HiddenMarkovModel, one_hot_decoding


class HiddenMarkovModelFilter(Filter):
    def __init__(self, system: HiddenMarkovModel) -> None:
        assert issubclass(type(system), HiddenMarkovModel)
        self._system = system
        super().__init__(
            state_dim=self._system.state_dim,
            observation_dim=self._system.observation_dim,
            number_of_systems=self._system.number_of_systems,
        )

    def _compute_estimation_process(self) -> NDArray[np.float64]:
        estimation_process: NDArray[np.float64] = self._estimation_process(
            state=self._estimated_state,
            observation=self._observation_history[:, :, 0],
            transition_probability_matrix=self._system.transition_probability_matrix,
            emission_probability_matrix=self._system.emission_probability_matrix,
        )
        return estimation_process

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _estimation_process(
        state: NDArray[np.float64],
        observation: NDArray[np.float64],
        transition_probability_matrix: NDArray[np.float64],
        emission_probability_matrix: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        estimated_state: NDArray[np.float64] = np.zeros_like(state)
        number_of_systems: int = state.shape[0]
        observation_value: NDArray[np.int64] = one_hot_decoding(observation)
        for i in range(number_of_systems):
            estimated_state[i, :] = (
                state[i, :]
                * emission_probability_matrix[:, observation_value[i]]
            ) @ transition_probability_matrix
            estimated_state[i, :] /= np.sum(estimated_state[i, :])
        return estimated_state
