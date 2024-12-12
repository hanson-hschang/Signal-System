from typing import Optional

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ss.system import DiscreteTimeSystem, SystemCallback
from ss.utility.assertion.validator import Validator


@njit(cache=True)  # type: ignore
def one_hot_encoding(
    values: NDArray[np.int64],
    basis: NDArray[np.float64],
) -> NDArray[np.float64]:
    return basis[values]


@njit(cache=True)  # type: ignore
def one_hot_decoding(
    one_hot_vectors: NDArray[np.float64],
) -> NDArray[np.int64]:
    number_of_systems = one_hot_vectors.shape[0]
    values = np.empty(number_of_systems, dtype=np.int64)
    for i in range(number_of_systems):
        values[i] = np.argmax(one_hot_vectors[i, :])
    return values


class HiddenMarkovModel(DiscreteTimeSystem):
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
                    f"transition_probability_matrix should have the shape as (state_dim, state_dim) with state_dim={self._state_dim}."
                    f"The transition_probability_matrix given has the shape {shape}."
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
        initial_distribution: Optional[ArrayLike] = None,
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

        if initial_distribution is None:
            initial_distribution = np.ones(self._state_dim) / self._state_dim
        initial_distribution = np.array(initial_distribution, dtype=np.float64)
        assert initial_distribution.shape[0] == self._state_dim, (
            f"initial_distribution should have the same length as state_dim {self._state_dim}."
            f"The initial_distribution given has shape {initial_distribution.shape}."
        )

        self._transition_probability_cumsum = np.cumsum(
            self._transition_probability_matrix, axis=1
        )
        self._emission_probability_cumsum = np.cumsum(
            self._emission_probability_matrix, axis=1
        )

        self._state_value: NDArray[np.int64] = np.random.choice(
            self._state_dim,
            size=self._number_of_systems,
            p=initial_distribution,
        )
        self._observation_value: NDArray[np.int64] = np.zeros(
            self._number_of_systems, dtype=np.int64
        )
        self._state_encoder_basis = np.identity(
            self._state_dim, dtype=np.float64
        )
        self._observation_encoder_basis = np.identity(
            self._observation_dim, dtype=np.float64
        )
        self._state[...] = one_hot_encoding(
            self._state_value,
            self._state_encoder_basis,
        )
        self.observe()

    @property
    def state_value(self) -> NDArray[np.int64]:
        return self._state_value.squeeze()

    @property
    def observation_value(self) -> NDArray[np.int64]:
        return self._observation_value.squeeze()

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

    def duplicate(self, number_of_systems: int) -> "HiddenMarkovModel":
        return HiddenMarkovModel(
            transition_probability_matrix=self._transition_probability_matrix,
            emission_probability_matrix=self._emission_probability_matrix,
            number_of_systems=number_of_systems,
        )

    def _compute_state_process(self) -> NDArray[np.float64]:
        self._state_value = self._process(
            self._state_value,
            self._transition_probability_cumsum,
        )
        # state_process is a one-hot embedding of the state_index
        state_process: NDArray[np.float64] = one_hot_encoding(
            self._state_value,
            self._state_encoder_basis,
        )
        return state_process

    def _compute_observation_process(self) -> NDArray[np.float64]:
        self._observation_value = self._process(
            self._state_value,
            self._emission_probability_cumsum,
        )
        # observation_process is a one-hot embedding of the observation_index
        observation_process: NDArray[np.float64] = one_hot_encoding(
            self._observation_value,
            self._observation_encoder_basis,
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
        system: HiddenMarkovModel,
    ) -> None:
        assert issubclass(
            type(system), HiddenMarkovModel
        ), f"system must be an instance of MarkovChain"
        super().__init__(step_skip, system)
        self._system: HiddenMarkovModel = system

    def _record(self, time: float) -> None:
        super()._record(time)
        self._callback_params["state_value"].append(
            self._system.state_value.copy()
        )
        self._callback_params["observation_value"].append(
            self._system.observation_value.copy()
        )
