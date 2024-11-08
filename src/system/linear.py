from typing import Any, Callable, Optional, Union, assert_never

from enum import StrEnum

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm

from assertion import Validator, isPositiveInteger
from system import DynamicSystem


class DiscreteTimeLinearSystem(DynamicSystem):
    class StateSpaceMatrixAValidator(Validator):
        def __init__(self, state_space_matrix_A: ArrayLike):
            super().__init__()
            self.state_space_matrix_A = np.array(state_space_matrix_A)
            self.validate_shape()

        def validate_shape(self) -> None:
            shape = self.state_space_matrix_A.shape
            if not (len(shape) == 2 and (shape[0] == shape[1])):
                self._errors.append(
                    "state_space_matrix_A should be a square matrix"
                )

    class StateSpaceMatrixBValidator(Validator):
        def __init__(self, state_space_matrix_B: ArrayLike, state_dim: int):
            super().__init__()
            self.state_space_matrix_B = np.array(state_space_matrix_B)
            self.state_dim = state_dim
            self.validate_shape()

        def validate_shape(self) -> None:
            shape = self.state_space_matrix_B.shape
            if not (len(shape) == 2 and (shape[0] == self.state_dim)):
                self._errors.append(
                    "state_space_matrix_A and state_space_matrix_B should share the same number of rows (state_dim)"
                )

    class StateSpaceMatrixCValidator(Validator):
        def __init__(self, state_space_matrix_C: ArrayLike, state_dim: int):
            super().__init__()
            self.state_space_matrix_C = np.array(state_space_matrix_C)
            self.state_dim = state_dim
            self.validate_shape()

        def validate_shape(self) -> None:
            shape = self.state_space_matrix_C.shape
            if not (len(shape) == 2 and (shape[1] == self.state_dim)):
                self._errors.append(
                    "state_space_matrix_A and state_space_matrix_C should share the same number of columns (state_dim)"
                )

    class ProcessNoiseCovarianceValidator(Validator):
        def __init__(self, process_noise_covariance: ArrayLike, state_dim: int):
            super().__init__()
            self.process_noise_covariance = np.array(process_noise_covariance)
            self.state_dim = state_dim
            self.validate_shape()

        def validate_shape(self) -> None:
            shape = self.process_noise_covariance.shape
            if not (
                len(shape) == 2 and (shape[0] == shape[1] == self.state_dim)
            ):
                self._errors.append(
                    "process_noise_covariance should be a square matrix and have shape (state_dim, state_dim)"
                )

    class ObservationNoiseCovarianceValidator(Validator):
        def __init__(
            self, observation_noise_covariance: ArrayLike, observation_dim: int
        ):
            super().__init__()
            self.observation_noise_covariance = np.array(
                observation_noise_covariance
            )
            self.observation_dim = observation_dim
            self.validate_shape()

        def validate_shape(self) -> None:
            shape = self.observation_noise_covariance.shape
            if not (
                len(shape) == 2
                and (shape[0] == shape[1] == self.observation_dim)
            ):
                self._errors.append(
                    "observation_noise_covariance should be a square matrix and have shape (observation_dim, observation_dim)"
                )

    def __init__(
        self,
        state_space_matrix_A: ArrayLike,
        state_space_matrix_C: ArrayLike,
        state_space_matrix_B: Optional[ArrayLike] = None,
        process_noise_covariance: Optional[ArrayLike] = None,
        observation_noise_covariance: Optional[ArrayLike] = None,
        number_of_systems: int = 1,
        time_step: float = 1.0,
    ):
        self.StateSpaceMatrixAValidator(state_space_matrix_A)
        state_space_matrix_A = np.array(state_space_matrix_A, dtype=np.float64)
        state_dim = state_space_matrix_A.shape[0]

        self.StateSpaceMatrixCValidator(state_space_matrix_C, state_dim)
        state_space_matrix_C = np.array(state_space_matrix_C, dtype=np.float64)
        observation_dim = state_space_matrix_C.shape[0]

        if state_space_matrix_B is not None:
            self.StateSpaceMatrixBValidator(state_space_matrix_B, state_dim)
            state_space_matrix_B = np.array(
                state_space_matrix_B, dtype=np.float64
            )
        else:
            state_space_matrix_B = np.zeros((state_dim, 0))
        control_dim = state_space_matrix_B.shape[1]
        self._set_compute_state_process(control_flag=(control_dim > 0))

        if process_noise_covariance is not None:
            self.ProcessNoiseCovarianceValidator(
                process_noise_covariance, state_dim
            )
            process_noise_covariance = np.array(
                process_noise_covariance, dtype=np.float64
            )
        else:
            process_noise_covariance = np.zeros((state_dim, state_dim))

        if observation_noise_covariance is not None:
            self.ObservationNoiseCovarianceValidator(
                observation_noise_covariance, observation_dim
            )
            observation_noise_covariance = np.array(
                observation_noise_covariance, dtype=np.float64
            )
        else:
            observation_noise_covariance = np.zeros(
                (observation_dim, observation_dim)
            )

        self.state_space_matrix_A = state_space_matrix_A
        self.state_space_matrix_B = state_space_matrix_B
        self.state_space_matrix_C = state_space_matrix_C
        self.process_noise_covariance = process_noise_covariance
        self.observation_noise_covariance = observation_noise_covariance

        super().__init__(
            state_dim=state_dim,
            observation_dim=observation_dim,
            control_dim=control_dim,
            number_of_systems=number_of_systems,
            time_step=time_step,
        )

    def _set_compute_state_process(self, control_flag: bool) -> None:
        def _compute_state_process_without_control() -> NDArray[np.float64]:
            state_process: NDArray[np.float64] = (
                self._state_process_without_control(
                    self._state,
                    self.state_space_matrix_A,
                )
            )
            return state_process

        def _compute_state_process_with_control() -> NDArray[np.float64]:
            state_process: NDArray[np.float64] = (
                self._state_process_with_control(
                    self._state,
                    self.state_space_matrix_A,
                    self._control,
                    self.state_space_matrix_B,
                )
            )
            return state_process

        self._compute_state_process: Callable[[], NDArray[np.float64]] = (
            _compute_state_process_with_control
            if control_flag
            else _compute_state_process_without_control
        )

    def update(self, time: Union[int, float]) -> Union[int, float]:
        """
        Update each system by one time step.
        """
        process_noise = np.random.multivariate_normal(
            np.zeros(self._state_dim),
            self.process_noise_covariance,
            size=self._number_of_systems,
        )
        self._update_state(
            self._state,
            self._compute_state_process(),
            process_noise,
        )
        return time + self._time_step

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _state_process_without_control(
        state: NDArray[np.float64],
        state_space_matrix_A: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        state_process = np.zeros_like(state)
        for i in range(state.shape[0]):
            state_process[i, :] = state_space_matrix_A @ state[i, :]
        return state_process

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _state_process_with_control(
        state: NDArray[np.float64],
        state_space_matrix_A: NDArray[np.float64],
        control: NDArray[np.float64],
        state_space_matrix_B: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        state_process_with_control = np.zeros_like(state)
        for i in range(state.shape[0]):
            state_process_with_control[i, :] = (
                state_space_matrix_A @ state[i, :]
                + state_space_matrix_B @ control[i, :]
            )
        return state_process_with_control

    def _compute_observation(self) -> None:
        observation_noise = np.random.multivariate_normal(
            np.zeros(self._observation_dim),
            self.observation_noise_covariance,
            size=self._number_of_systems,
        )
        self._compute_observation_process(
            self._state,
            self._observation,
            self.state_space_matrix_C,
            observation_noise,
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _compute_observation_process(
        state: NDArray[np.float64],
        observation: NDArray[np.float64],
        state_space_matrix_C: NDArray[np.float64],
        noise: NDArray[np.float64],
    ) -> None:
        for i in range(state.shape[0]):
            observation[i, :] = state_space_matrix_C @ state[i, :] + noise[i, :]
