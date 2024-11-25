from typing import Optional

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ss.system.state_vector import SystemCallback
from ss.system.state_vector.dynamic_system import DiscreteTimeSystem
from ss.tool.assertion.validator import Validator


@njit(cache=True)  # type: ignore
def _one_hot_embedding(
    indices: NDArray[np.int64],
    identity: NDArray[np.float64],
) -> NDArray[np.float64]:
    return identity[indices]


class MarkovChain(DiscreteTimeSystem):
    class _TransitionProbabilityMatrixValidator(Validator):
        def __init__(
            self,
            transition_probability_matrix: ArrayLike,
            state_dim: Optional[int] = None,
        ) -> None:
            super().__init__()
            self._transition_probability_matrix = np.array(
                transition_probability_matrix, dtype=np.float64
            )
            self._state_dim = state_dim
            self._validate_shape()
            self._validate_row_sum()

        def _validate_shape(self) -> None:
            shape = self._transition_probability_matrix.shape
            if not (len(shape) == 2 and (shape[0] == shape[1])):
                self._errors.append(
                    "transition_probability_matrix should be a square matrix"
                )
            if self._state_dim is not None and shape[0] != self._state_dim:
                self._errors.append(
                    f"transition_probability_matrix should have the shape as (state_dim, state_dim) with state_dim={self._state_dim}. Got {shape}"
                )

        def _validate_row_sum(self) -> None:
            row_sum = np.sum(self._transition_probability_matrix, axis=1)
            if not np.allclose(row_sum, np.ones_like(row_sum)):
                self._errors.append(
                    "transition_probability_matrix should have row sum equal to 1"
                )

        def get_matrix(self) -> NDArray[np.float64]:
            return self._transition_probability_matrix

    class _EmissionProbabilityMatrixValidator(Validator):
        def __init__(
            self, state_dim: int, emission_probability_matrix: ArrayLike
        ) -> None:
            super().__init__()
            self._emission_probability_matrix = np.array(
                emission_probability_matrix, dtype=np.float64
            )
            self._state_dim = state_dim
            self._validate_shape()
            self._validate_row_sum()

        def _validate_shape(self) -> None:
            shape = self._emission_probability_matrix.shape
            if not (len(shape) == 2 and (shape[0] == self._state_dim)):
                self._errors.append(
                    "transition_probability_matrix and emission_probability_matrix should have the same number of rows (state_dim)"
                )

        def _validate_row_sum(self) -> None:
            row_sum = np.sum(self._emission_probability_matrix, axis=1)
            if not np.allclose(row_sum, np.ones_like(row_sum)):
                self._errors.append(
                    "emission_probability_matrix should have row sum equal to 1"
                )

        def get_matrix(self) -> NDArray[np.float64]:
            return self._emission_probability_matrix

    def __init__(
        self,
        transition_probability_matrix: ArrayLike,
        emission_probability_matrix: ArrayLike,
        number_of_systems: int = 1,
    ) -> None:
        self._transition_probability_matrix = (
            self._TransitionProbabilityMatrixValidator(
                transition_probability_matrix
            ).get_matrix()
        )
        state_dim = self._transition_probability_matrix.shape[0]

        self._emission_probability_matrix = (
            self._EmissionProbabilityMatrixValidator(
                state_dim=state_dim,
                emission_probability_matrix=emission_probability_matrix,
            ).get_matrix()
        )
        observation_dim = self._emission_probability_matrix.shape[1]

        super().__init__(
            state_dim=state_dim,
            observation_dim=observation_dim,
            number_of_systems=number_of_systems,
        )

        self._transition_probability_cumsum = np.cumsum(
            self._transition_probability_matrix, axis=1
        )
        self._emission_probability_cumsum = np.cumsum(
            self._emission_probability_matrix, axis=1
        )
        self._state_index: NDArray[np.int64] = np.zeros(
            self._number_of_systems, dtype=np.int64
        )
        self._observation_index: NDArray[np.int64] = np.zeros(
            self._number_of_systems, dtype=np.int64
        )
        self._state_embedding = np.identity(self._state_dim, dtype=np.float64)
        self._observation_embedding = np.identity(
            self._observation_dim, dtype=np.float64
        )
        self._state[...] = _one_hot_embedding(
            self._state_index,
            self._state_embedding,
        )

    @property
    def state_index(self) -> NDArray[np.int64]:
        return self._state_index.squeeze()

    @property
    def observation_index(self) -> NDArray[np.int64]:
        return self._observation_index.squeeze()

    @property
    def transition_probability_matrix(self) -> NDArray[np.float64]:
        return self._transition_probability_matrix

    @transition_probability_matrix.setter
    def transition_probability_matrix(self, matrix: ArrayLike) -> None:
        self._transition_probability_matrix = (
            self._TransitionProbabilityMatrixValidator(
                matrix,
                state_dim=self.state_dim,
            ).get_matrix()
        )
        self._transition_probability_cumsum = np.cumsum(
            self._transition_probability_matrix, axis=1
        )

    @property
    def emission_probability_matrix(self) -> NDArray[np.float64]:
        return self._emission_probability_matrix

    @emission_probability_matrix.setter
    def emission_probability_matrix(self, matrix: ArrayLike) -> None:
        self._emission_probability_matrix = (
            self._EmissionProbabilityMatrixValidator(
                state_dim=self.state_dim,
                emission_probability_matrix=matrix,
            ).get_matrix()
        )
        self._emission_probability_cumsum = np.cumsum(
            self._emission_probability_matrix, axis=1
        )

    def create_multiple_systems(self, number_of_systems: int) -> "MarkovChain":
        return MarkovChain(
            transition_probability_matrix=self._transition_probability_matrix,
            emission_probability_matrix=self._emission_probability_matrix,
            number_of_systems=number_of_systems,
        )

    def _compute_state_process(self) -> NDArray[np.float64]:
        self._state_index = self._process(
            self._state_index,
            self._transition_probability_cumsum,
        )
        # state_process is a one-hot embedding of the state_index
        state_process: NDArray[np.float64] = _one_hot_embedding(
            self._state_index,
            self._state_embedding,
        )
        return state_process

    def _compute_observation_process(self) -> NDArray[np.float64]:
        self._observation_index: NDArray[np.int64] = self._process(
            self._state_index,
            self._emission_probability_cumsum,
        )
        # observation_process is a one-hot embedding of the observation_index
        observation_process: NDArray[np.float64] = _one_hot_embedding(
            self._observation_index,
            self._observation_embedding,
        )
        return observation_process

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _process(
        input_index: NDArray[np.int64],
        probability_cumsum_matrix: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        number_of_systems = input_index.shape[0]
        random_numbers = np.random.rand(number_of_systems)
        output_index = np.empty(number_of_systems, dtype=np.int64)
        for i, random_number in enumerate(random_numbers):
            output_index[i] = np.searchsorted(
                probability_cumsum_matrix[input_index[i], :],
                random_number,
                side="right",
            )
        return output_index


class MarkovChainCallback(SystemCallback):
    def __init__(
        self,
        step_skip: int,
        system: MarkovChain,
    ) -> None:
        assert issubclass(
            type(system), MarkovChain
        ), f"system must be an instance of MarkovChain"
        super().__init__(step_skip, system)
        self._system: MarkovChain = system

    def _record_params(self, time: float) -> None:
        super()._record_params(time)
        self._callback_params["state_index"].append(
            self._system.state_index.copy()
        )
        self._callback_params["observation_index"].append(
            self._system.observation_index.copy()
        )
