from typing import Callable, Optional, Union

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm

from ss.system.state_vector.dynamic_system import DiscreteTimeSystem
from ss.tool.assertion import isPositiveNumber
from ss.tool.assertion.validator import Validator


class DiscreteTimeLinearSystem(DiscreteTimeSystem):
    class _StateSpaceMatrixAValidator(Validator):
        def __init__(self, state_space_matrix_A: ArrayLike) -> None:
            super().__init__()
            self._state_space_matrix_A = np.array(
                state_space_matrix_A, dtype=np.float64
            )
            self._validate_shape()

        def _validate_shape(self) -> None:
            shape = self._state_space_matrix_A.shape
            if not (len(shape) == 2 and (shape[0] == shape[1])):
                self._errors.append(
                    "state_space_matrix_A should be a square matrix"
                )

        def get_matrix(self) -> NDArray[np.float64]:
            return self._state_space_matrix_A

    class _StateSpaceMatrixBValidator(Validator):
        def __init__(
            self,
            state_dim: int,
            state_space_matrix_B: Optional[ArrayLike] = None,
        ) -> None:
            super().__init__()
            if state_space_matrix_B is None:
                state_space_matrix_B = np.zeros((state_dim, 0))
            self._state_space_matrix_B = np.array(
                state_space_matrix_B, dtype=np.float64
            )
            self._state_dim = state_dim
            self._validate_shape()

        def _validate_shape(self) -> None:
            shape = self._state_space_matrix_B.shape
            if not (len(shape) == 2 and (shape[0] == self._state_dim)):
                self._errors.append(
                    "state_space_matrix_A and state_space_matrix_B should share the same number of rows (state_dim)"
                )

        def get_matrix(self) -> NDArray[np.float64]:
            return self._state_space_matrix_B

    class _StateSpaceMatrixCValidator(Validator):
        def __init__(
            self, state_dim: int, state_space_matrix_C: ArrayLike
        ) -> None:
            super().__init__()
            self._state_space_matrix_C = np.array(
                state_space_matrix_C, dtype=np.float64
            )
            self._state_dim = state_dim
            self._validate_shape()

        def _validate_shape(self) -> None:
            shape = self._state_space_matrix_C.shape
            if not (len(shape) == 2 and (shape[1] == self._state_dim)):
                self._errors.append(
                    "state_space_matrix_A and state_space_matrix_C should share the same number of columns (state_dim)"
                )

        def get_matrix(self) -> NDArray[np.float64]:
            return self._state_space_matrix_C

    def __init__(
        self,
        state_space_matrix_A: ArrayLike,
        state_space_matrix_C: ArrayLike,
        state_space_matrix_B: Optional[ArrayLike] = None,
        process_noise_covariance: Optional[ArrayLike] = None,
        observation_noise_covariance: Optional[ArrayLike] = None,
        number_of_systems: int = 1,
    ) -> None:
        self._state_space_matrix_A = self._StateSpaceMatrixAValidator(
            state_space_matrix_A
        ).get_matrix()
        state_dim = self._state_space_matrix_A.shape[0]

        self._state_space_matrix_C = self._StateSpaceMatrixCValidator(
            state_dim, state_space_matrix_C
        ).get_matrix()
        observation_dim = self._state_space_matrix_C.shape[0]

        self._state_space_matrix_B = self._StateSpaceMatrixBValidator(
            state_dim, state_space_matrix_B
        ).get_matrix()
        control_dim = self._state_space_matrix_B.shape[1]
        self._set_compute_state_process(control_flag=(control_dim > 0))

        super().__init__(
            state_dim=state_dim,
            observation_dim=observation_dim,
            control_dim=control_dim,
            number_of_systems=number_of_systems,
            process_noise_covariance=process_noise_covariance,
            observation_noise_covariance=observation_noise_covariance,
        )

    @property
    def state_space_matrix_A(self) -> NDArray[np.float64]:
        return self._state_space_matrix_A

    @property
    def state_space_matrix_B(self) -> NDArray[np.float64]:
        return self._state_space_matrix_B

    @property
    def state_space_matrix_C(self) -> NDArray[np.float64]:
        return self._state_space_matrix_C

    def create_multiple_systems(
        self, number_of_systems: int
    ) -> "DiscreteTimeLinearSystem":
        """
        Create multiple systems based on the current system.

        Parameters
        ----------
        `number_of_systems: int`
            The number of systems to be created.

        Returns
        -------
        `system: DiscreteTimeLinearSystem`
            The created multi-system.
        """
        return self.__class__(
            state_space_matrix_A=self._state_space_matrix_A,
            state_space_matrix_C=self._state_space_matrix_C,
            state_space_matrix_B=self._state_space_matrix_B,
            process_noise_covariance=self._process_noise_covariance,
            observation_noise_covariance=self._observation_noise_covariance,
            number_of_systems=number_of_systems,
        )

    def _set_compute_state_process(self, control_flag: bool) -> None:
        def _compute_state_process_without_control() -> NDArray[np.float64]:
            state_process: NDArray[np.float64] = (
                self._state_process_without_control(
                    self._state,
                    self._state_space_matrix_A,
                )
            )
            return state_process

        def _compute_state_process_with_control() -> NDArray[np.float64]:
            state_process: NDArray[np.float64] = (
                self._state_process_with_control(
                    self._state,
                    self._state_space_matrix_A,
                    self._control,
                    self._state_space_matrix_B,
                )
            )
            return state_process

        if control_flag:
            setattr(
                self,
                "_compute_state_process",
                _compute_state_process_with_control,
            )
        else:
            setattr(
                self,
                "_compute_state_process",
                _compute_state_process_without_control,
            )

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

    def _compute_observation_process(self) -> NDArray[np.float64]:
        observation_process: NDArray[np.float64] = self._observation_process(
            self._state,
            self._state_space_matrix_C,
        )
        return observation_process

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _observation_process(
        state: NDArray[np.float64],
        state_space_matrix_C: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        observation = np.zeros((state.shape[0], state_space_matrix_C.shape[0]))
        for i in range(state.shape[0]):
            observation[i, :] = state_space_matrix_C @ state[i, :]
        return observation


class ContinuousTimeLinearSystem(DiscreteTimeLinearSystem):
    def __init__(
        self,
        time_step: Union[int, float],
        state_space_matrix_A: ArrayLike,
        state_space_matrix_C: ArrayLike,
        state_space_matrix_B: Optional[ArrayLike] = None,
        process_noise_covariance: Optional[ArrayLike] = None,
        observation_noise_covariance: Optional[ArrayLike] = None,
        number_of_systems: int = 1,
    ) -> None:
        assert isPositiveNumber(
            time_step
        ), f"time_step {time_step} must be a positive number"
        super().__init__(
            state_space_matrix_A=state_space_matrix_A,
            state_space_matrix_C=state_space_matrix_C,
            state_space_matrix_B=state_space_matrix_B,
            process_noise_covariance=process_noise_covariance,
            observation_noise_covariance=observation_noise_covariance,
            number_of_systems=number_of_systems,
        )
        self._time_step = time_step

        self._continuous_time_state_space_matrix_A = (
            self._state_space_matrix_A.copy()
        )
        self._continuous_time_state_space_matrix_B = (
            self._state_space_matrix_B.copy()
        )
        self._continuous_time_state_space_matrix_C = (
            self._state_space_matrix_C.copy()
        )

        self._state_space_matrix_A = np.array(
            expm(self._state_space_matrix_A * self._time_step), dtype=np.float64
        )
        self._state_space_matrix_B = (
            self._state_space_matrix_A
            @ self._state_space_matrix_B
            * self._time_step
        )

    @property
    def state_space_matrix_A(self) -> NDArray[np.float64]:
        return self._continuous_time_state_space_matrix_A

    @property
    def state_space_matrix_B(self) -> NDArray[np.float64]:
        return self._continuous_time_state_space_matrix_B

    @property
    def state_space_matrix_C(self) -> NDArray[np.float64]:
        return self._continuous_time_state_space_matrix_C
