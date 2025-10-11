import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm

from ss.system import DiscreteTimeSystem
from ss.utility.assertion import is_positive_number
from ss.utility.assertion.validator import Validator


class DiscreteTimeLinearSystem(DiscreteTimeSystem):
    class _StateSpaceMatrixAValidator(Validator):
        def __init__(self, state_space_matrix_A: ArrayLike) -> None:
            super().__init__(state_space_matrix_A)
            self._state_space_matrix_A = np.array(state_space_matrix_A)
            self._validate_functions.append(self._validate_shape)

        def _validate_shape(self) -> bool:
            shape = self._state_space_matrix_A.shape
            if (len(shape) == 2) and (shape[0] == shape[1]):
                return True
            self._errors.append(
                "state_space_matrix_A should be a square matrix"
            )
            return False

        def get_matrix(self) -> NDArray:
            return self._state_space_matrix_A

    class _StateSpaceMatrixBValidator(Validator):
        def __init__(
            self,
            state_space_matrix_B: ArrayLike | None,
            state_dim: int,
        ) -> None:
            super().__init__(state_space_matrix_B)
            if state_space_matrix_B is None:
                state_space_matrix_B = np.zeros((state_dim, 0))
            self._state_space_matrix_B = np.array(state_space_matrix_B)
            self._state_dim = state_dim
            self._validate_functions.append(self._validate_shape)

        def _validate_shape(self) -> bool:
            shape = self._state_space_matrix_B.shape
            if (len(shape) == 2) and (shape[0] == self._state_dim):
                return True
            self._errors.append(
                "state_space_matrix_A and state_space_matrix_B should "
                "share the same number of rows (state_dim)"
                f" with state_dim={self._state_dim}. "
                f"Got state_space_matrix_B.shape={shape}."
            )
            return False

        def get_matrix(self) -> NDArray:
            return self._state_space_matrix_B

    class _StateSpaceMatrixCValidator(Validator):
        def __init__(
            self,
            state_space_matrix_C: ArrayLike,
            state_dim: int,
        ) -> None:
            super().__init__(state_space_matrix_C)
            self._state_space_matrix_C = np.array(state_space_matrix_C)
            self._state_dim = state_dim
            self._validate_functions.append(self._validate_shape)

        def _validate_shape(self) -> bool:
            shape = self._state_space_matrix_C.shape
            if (len(shape) == 2) and (shape[1] == self._state_dim):
                return True
            self._errors.append(
                "state_space_matrix_A and state_space_matrix_C "
                "should share the same number of columns (state_dim)"
                f" with state_dim={self._state_dim}. "
                f"Got state_space_matrix_C.shape={shape}."
            )
            return False

        def get_matrix(self) -> NDArray:
            return self._state_space_matrix_C

    def __init__(
        self,
        state_space_matrix_A: ArrayLike,
        state_space_matrix_C: ArrayLike,
        state_space_matrix_B: ArrayLike | None = None,
        process_noise_covariance: ArrayLike | None = None,
        observation_noise_covariance: ArrayLike | None = None,
        batch_size: int = 1,
    ) -> None:
        self._state_space_matrix_A = self._StateSpaceMatrixAValidator(
            state_space_matrix_A
        ).get_matrix()
        state_dim = self._state_space_matrix_A.shape[0]

        self._state_space_matrix_C = self._StateSpaceMatrixCValidator(
            state_space_matrix_C,
            state_dim,
        ).get_matrix()
        observation_dim = self._state_space_matrix_C.shape[0]

        self._state_space_matrix_B = self._StateSpaceMatrixBValidator(
            state_space_matrix_B,
            state_dim,
        ).get_matrix()
        control_dim = self._state_space_matrix_B.shape[1]
        self._set_compute_state_process(control_flag=(control_dim > 0))

        super().__init__(
            state_dim=state_dim,
            observation_dim=observation_dim,
            control_dim=control_dim,
            batch_size=batch_size,
            process_noise_covariance=process_noise_covariance,
            observation_noise_covariance=observation_noise_covariance,
        )

    @property
    def state_space_matrix_A(self) -> NDArray:
        return self._state_space_matrix_A

    @property
    def state_space_matrix_B(self) -> NDArray:
        return self._state_space_matrix_B

    @property
    def state_space_matrix_C(self) -> NDArray:
        return self._state_space_matrix_C

    def duplicate(self, batch_size: int) -> "DiscreteTimeLinearSystem":
        """
        Create multiple systems based on the current system.

        Parameters
        ----------
        `batch_size: int`
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
            batch_size=batch_size,
        )

    def _set_compute_state_process(self, control_flag: bool) -> None:
        def _compute_state_process_without_control() -> NDArray:
            state_process: NDArray = self._state_process_without_control(
                self._state,
                self._state_space_matrix_A,
            )
            return state_process

        def _compute_state_process_with_control() -> NDArray:
            state_process: NDArray = self._state_process_with_control(
                self._state,
                self._state_space_matrix_A,
                self._control,
                self._state_space_matrix_B,
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
        state: NDArray,
        state_space_matrix_A: NDArray,
    ) -> NDArray:
        state_process = np.zeros_like(state)
        for i in range(state.shape[0]):
            state_process[i, :] = state_space_matrix_A @ state[i, :]
        return state_process

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _state_process_with_control(
        state: NDArray,
        state_space_matrix_A: NDArray,
        control: NDArray,
        state_space_matrix_B: NDArray,
    ) -> NDArray:
        state_process_with_control = np.zeros_like(state)
        for i in range(state.shape[0]):
            state_process_with_control[i, :] = (
                state_space_matrix_A @ state[i, :]
                + state_space_matrix_B @ control[i, :]
            )
        return state_process_with_control

    def _compute_observation_process(self) -> NDArray:
        observation_process: NDArray = self._observation_process(
            self._state,
            self._state_space_matrix_C,
        )
        return observation_process

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _observation_process(
        state: NDArray,
        state_space_matrix_C: NDArray,
    ) -> NDArray:
        observation = np.zeros((state.shape[0], state_space_matrix_C.shape[0]))
        for i in range(state.shape[0]):
            observation[i, :] = state_space_matrix_C @ state[i, :]
        return observation


class ContinuousTimeLinearSystem(DiscreteTimeLinearSystem):
    def __init__(
        self,
        time_step: int | float,
        state_space_matrix_A: ArrayLike,
        state_space_matrix_C: ArrayLike,
        state_space_matrix_B: ArrayLike | None = None,
        process_noise_covariance: ArrayLike | None = None,
        observation_noise_covariance: ArrayLike | None = None,
        batch_size: int = 1,
    ) -> None:
        assert is_positive_number(time_step), (
            f"time_step {time_step} must be a positive number"
        )
        super().__init__(
            state_space_matrix_A=state_space_matrix_A,
            state_space_matrix_C=state_space_matrix_C,
            state_space_matrix_B=state_space_matrix_B,
            process_noise_covariance=process_noise_covariance,
            observation_noise_covariance=observation_noise_covariance,
            batch_size=batch_size,
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
            expm(self._state_space_matrix_A * self._time_step)
        )
        self._state_space_matrix_B = (
            self._state_space_matrix_A
            @ self._state_space_matrix_B
            * self._time_step
        )

    @property
    def state_space_matrix_A(self) -> NDArray:
        return self._continuous_time_state_space_matrix_A

    @property
    def state_space_matrix_B(self) -> NDArray:
        return self._continuous_time_state_space_matrix_B

    @property
    def state_space_matrix_C(self) -> NDArray:
        return self._continuous_time_state_space_matrix_C
