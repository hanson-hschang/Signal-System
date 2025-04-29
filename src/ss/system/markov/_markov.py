from typing import Optional, Tuple, Union, cast

from pathlib import Path

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ss.system import DiscreteTimeSystem, SystemCallback
from ss.utility.assertion.validator import Validator
from ss.utility.descriptor import ReadOnlyDescriptor


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
    class _TransitionMatrixValidator(Validator):
        def __init__(
            self,
            transition_matrix: ArrayLike,
            discrete_state_dim: Optional[int] = None,
        ) -> None:
            super().__init__(transition_matrix)
            self._transition_matrix = np.array(
                transition_matrix, dtype=np.float64
            )
            self._discrete_state_dim = discrete_state_dim
            self._validate_functions.append(self._validate_shape)
            self._validate_functions.append(self._validate_row_sum)

        def _validate_shape(self) -> bool:
            shape = self._transition_matrix.shape
            is_valid = True
            if not (len(shape) == 2 and (shape[0] == shape[1])):
                self._errors.append(
                    "transition_matrix should be a square matrix"
                )
                is_valid = False
            if (
                self._discrete_state_dim is not None
                and shape[0] != self._discrete_state_dim
            ):
                self._errors.append(
                    f"transition_matrix should have the shape as (discrete_state_dim, discrete_state_dim) "
                    f"with discrete_state_dim={self._discrete_state_dim}. "
                    f"The transition_matrix given has the shape {shape}."
                )
                is_valid = False
            return is_valid

        def _validate_row_sum(self) -> bool:
            row_sum = np.sum(self._transition_matrix, axis=1)
            if np.allclose(row_sum, np.ones_like(row_sum)):
                return True
            self._errors.append(
                "transition_matrix should have row sum equal to 1"
            )
            return False

        def get_matrices(self) -> Tuple[NDArray, NDArray]:
            transition_cumsum_matrix = np.cumsum(
                self._transition_matrix, axis=1
            )
            return (
                self._transition_matrix,
                transition_cumsum_matrix,
            )

    class _EmissionMatrixValidator(Validator):
        def __init__(
            self,
            emission_matrix: ArrayLike,
            discrete_state_dim: int,
        ) -> None:
            super().__init__(emission_matrix)
            self._emission_matrix = np.array(emission_matrix, dtype=np.float64)
            self._discrete_state_dim = discrete_state_dim
            self._validate_functions.append(self._validate_shape)
            self._validate_functions.append(self._validate_row_sum)

        def _validate_shape(self) -> bool:
            shape = self._emission_matrix.shape
            if len(shape) == 2 and (shape[0] == self._discrete_state_dim):
                return True
            self._errors.append(
                "transition_matrix and emission_matrix should have the same number of rows (discrete_state_dim)"
            )
            return False

        def _validate_row_sum(self) -> bool:
            row_sum = np.sum(self._emission_matrix, axis=1)
            if np.allclose(row_sum, np.ones_like(row_sum)):
                return True
            self._errors.append(
                "emission_matrix should have row sum equal to 1"
            )
            return False

        def get_matrices(self) -> Tuple[NDArray, NDArray]:
            emission_cumsum_matrix = np.cumsum(self._emission_matrix, axis=1)
            return (
                self._emission_matrix,
                emission_cumsum_matrix,
            )

    def __init__(
        self,
        transition_matrix: ArrayLike,
        emission_matrix: ArrayLike,
        initial_distribution: Optional[ArrayLike] = None,
        number_of_systems: int = 1,
    ) -> None:
        (
            self._transition_matrix,
            self._transition_cumsum_matrix,
        ) = self._TransitionMatrixValidator(transition_matrix).get_matrices()
        self._discrete_state_dim = self._transition_matrix.shape[0]

        (
            self._emission_matrix,
            self._emission_cumsum_matrix,
        ) = self._EmissionMatrixValidator(
            emission_matrix=emission_matrix,
            discrete_state_dim=self._discrete_state_dim,
        ).get_matrices()
        self._discrete_observation_dim = self._emission_matrix.shape[1]

        super().__init__(
            state_dim=1,
            observation_dim=1,
            number_of_systems=number_of_systems,
        )

        if initial_distribution is None:
            initial_distribution = (
                np.ones(self._discrete_state_dim) / self._discrete_state_dim
            )
        self._initial_distribution = np.array(
            initial_distribution, dtype=np.float64
        )
        assert (
            self._initial_distribution.shape[0] == self._discrete_state_dim
        ), (
            f"initial_distribution should have the same length as discrete_state_dim {self._discrete_state_dim}. "
            f"The initial_distribution given has shape {self._initial_distribution.shape}."
        )

        self._state[:, :] = np.random.choice(
            self._discrete_state_dim,
            size=(self._number_of_systems, self._state_dim),
            p=self._initial_distribution,
        ).astype(
            np.float64
        )  # (number_of_systems, 1)
        self._state_encoder_basis = np.identity(
            self._discrete_state_dim, dtype=np.float64
        )
        self._observation_encoder_basis = np.identity(
            self._discrete_observation_dim, dtype=np.float64
        )
        self.observe()

    discrete_state_dim = ReadOnlyDescriptor[int]()
    discrete_observation_dim = ReadOnlyDescriptor[int]()

    @property
    def state_one_hot(self) -> NDArray:
        state_one_hot: NDArray = one_hot_encoding(
            self._state[:, 0].astype(dtype=np.int64),
            self._state_encoder_basis,
        )  # (number_of_systems, discrete_state_dim)
        if self._number_of_systems == 1:
            return state_one_hot[0, :]
        return state_one_hot

    @property
    def observation_one_hot(self) -> NDArray:
        observation_one_hot: NDArray = one_hot_encoding(
            self._observation[:, 0].astype(dtype=np.int64),
            self._observation_encoder_basis,
        )  # (number_of_systems, discrete_observation_dim)
        if self._number_of_systems == 1:
            return observation_one_hot[0, :]
        return observation_one_hot

    @property
    def transition_matrix(self) -> NDArray:
        return self._transition_matrix

    @transition_matrix.setter
    def transition_matrix(self, matrix: ArrayLike) -> None:
        (
            self._transition_matrix,
            self._transition_cumsum_matrix,
        ) = self._TransitionMatrixValidator(
            matrix,
            discrete_state_dim=self._discrete_state_dim,
        ).get_matrices()

    @property
    def emission_matrix(self) -> NDArray:
        return self._emission_matrix

    @emission_matrix.setter
    def emission_matrix(self, matrix: ArrayLike) -> None:
        (
            self._emission_matrix,
            self._emission_cumsum_matrix,
        ) = self._EmissionMatrixValidator(
            emission_matrix=matrix,
            discrete_state_dim=self._discrete_state_dim,
        ).get_matrices()

    def duplicate(self, number_of_systems: int) -> "HiddenMarkovModel":
        return HiddenMarkovModel(
            transition_matrix=self._transition_matrix,
            emission_matrix=self._emission_matrix,
            number_of_systems=number_of_systems,
        )

    def _compute_state_process(self) -> NDArray:
        state: NDArray = self._process(
            self._state.astype(np.int64),
            self._transition_cumsum_matrix,
        )  # (number_of_systems, 1)
        return state.astype(np.float64)

    def _compute_observation_process(self) -> NDArray:
        observation: NDArray = self._process(
            self._state.astype(np.int64),
            self._emission_cumsum_matrix,
        )  # (number_of_systems, 1)
        return observation.astype(np.float64)

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _process(
        input_index: NDArray[np.int64],
        probability_cumsum_matrix: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        number_of_systems = input_index.shape[0]
        random_numbers: NDArray = np.random.rand(number_of_systems)
        output_index = np.empty((number_of_systems, 1), dtype=np.int64)
        for i, random_number in enumerate(random_numbers):
            output_index[i, 0] = np.searchsorted(
                probability_cumsum_matrix[input_index[i, 0], :],
                random_number,
                side="right",
            )
        return output_index


class HmmCallback(SystemCallback[HiddenMarkovModel]):
    def __init__(
        self,
        step_skip: int,
        system: HiddenMarkovModel,
    ) -> None:
        assert issubclass(
            type(system), HiddenMarkovModel
        ), f"system must be a subclass of {HiddenMarkovModel.__name__}"
        super().__init__(step_skip, system)

    def save(self, filename: Union[str, Path]) -> None:
        self.add_meta_data(
            transition_matrix=self._system.transition_matrix,
            emission_matrix=self._system.emission_matrix,
        )
        self.add_meta_info(
            discrete_state_dim=self._system.discrete_state_dim,
            discrete_observation_dim=self._system.discrete_observation_dim,
        )
        super().save(filename)
