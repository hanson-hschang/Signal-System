from typing import Any, Literal, Optional, get_args

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm

from assertion import isPositiveInteger
from system import DynamicMixin, System


class DiscreteTimeLinearSystem(DynamicMixin, System):
    def __init__(
        self,
        state_space_matrix_A: ArrayLike,
        state_space_matrix_C: ArrayLike,
        state_space_matrix_B: Optional[ArrayLike] = None,
        process_noise_covariance_Q: Optional[ArrayLike] = None,
        observation_noise_covariance_R: Optional[ArrayLike] = None,
        number_of_systems: int = 1,
    ):
        state_space_matrix_A = np.array(state_space_matrix_A, dtype=np.float64)
        state_space_matrix_C = np.array(state_space_matrix_C, dtype=np.float64)

        assert (len(state_space_matrix_A.shape) == 2) and (
            state_space_matrix_A.shape[0] == state_space_matrix_A.shape[1]
        ), "state_space_matrix_A should be a square matrix"
        assert (
            len(state_space_matrix_C.shape) == 2
        ), "state_space_matrix_C should be a 2D matrix"
        assert (
            state_space_matrix_A.shape[1] == state_space_matrix_C.shape[1]
        ), "state_space_matrix_A and state_space_matrix_C should have compatible shapes"

        if state_space_matrix_B is not None:
            state_space_matrix_B = np.array(
                state_space_matrix_B, dtype=np.float64
            )
            assert (
                len(state_space_matrix_B.shape) == 2
            ), "state_space_matrix_B should be a 2D matrix"
            assert (
                state_space_matrix_A.shape[0] == state_space_matrix_B.shape[0]
            ), "state_space_matrix_A and state_space_matrix_B should have compatible shapes"
            self._update_state = self._update_state_with_control
        else:
            state_space_matrix_B = np.zeros((state_space_matrix_A.shape[0], 0))
            self._update_state = self._update_state_without_control

        if process_noise_covariance_Q is not None:
            process_noise_covariance_Q = np.array(
                process_noise_covariance_Q, dtype=np.float64
            )
            assert (len(process_noise_covariance_Q.shape) == 2) and (
                process_noise_covariance_Q.shape[0]
                == process_noise_covariance_Q.shape[1]
            ), "process_noise_covariance_Q should be a square matrix"
            assert (
                process_noise_covariance_Q.shape[0]
                == state_space_matrix_A.shape[0]
            ), "process_noise_covariance_Q and state_space_matrix_A should have compatible shapes"
        else:
            process_noise_covariance_Q = np.zeros(state_space_matrix_A.shape)

        if observation_noise_covariance_R is not None:
            observation_noise_covariance_R = np.array(
                observation_noise_covariance_R, dtype=np.float64
            )
            assert (len(observation_noise_covariance_R.shape) == 2) and (
                observation_noise_covariance_R.shape[0]
                == observation_noise_covariance_R.shape[1]
            ), "observation_noise_covariance_R should be a square matrix"
            assert (
                observation_noise_covariance_R.shape[0]
                == state_space_matrix_C.shape[0]
            ), "observation_noise_covariance_R and state_space_matrix_C should have compatible shapes"
        else:
            observation_noise_covariance_R = np.zeros(
                (state_space_matrix_C.shape[0], state_space_matrix_C.shape[0])
            )

        self._state_space_matrix_A = state_space_matrix_A
        self._state_space_matrix_B = state_space_matrix_B
        self._state_space_matrix_C = state_space_matrix_C
        self._process_noise_covariance_Q = process_noise_covariance_Q
        self._observation_noise_covariance_R = observation_noise_covariance_R

        super().__init__(
            state_dim=state_space_matrix_A.shape[0],
            observation_dim=state_space_matrix_C.shape[0],
            control_dim=state_space_matrix_B.shape[1],
            number_of_systems=number_of_systems,
            time_step=1,
        )

    @property
    def state_space_matrix_A(self) -> NDArray[np.float64]:
        """
        `state_space_matrix_A: ArrayLike[float]`
            The state space matrix of the system. Shape of the matrix is `(state_dim, state_dim)`.
        """
        return self._state_space_matrix_A

    @property
    def state_space_matrix_B(self) -> NDArray[np.float64]:
        """
        `state_space_matrix_B: ArrayLike[float]`
            The control matrix of the system. Shape of the matrix is `(state_dim, control_dim)`.
        """
        return self._state_space_matrix_B

    @property
    def state_space_matrix_C(self) -> NDArray[np.float64]:
        """
        `state_space_matrix_C: ArrayLike[float]`
            The observation matrix of the system. Shape of the matrix is `(observation_dim, state_dim)`.
        """
        return self._state_space_matrix_C

    @property
    def process_noise_covariance_Q(self) -> NDArray[np.float64]:
        """
        `process_noise_covariance_Q: ArrayLike[float]`
            The process noise covariance matrix of the system. Shape of the matrix is `(state_dim, state_dim)`.
        """
        return self._process_noise_covariance_Q

    @property
    def observation_noise_covariance_R(self) -> NDArray[np.float64]:
        """
        `observation_noise_covariance_R: ArrayLike[float]`
            The observation noise covariance matrix of the system. Shape of the matrix is `(observation_dim, observation_dim)`.
        """
        return self._observation_noise_covariance_R

    def update(self) -> None:
        """
        Update the state vector using the state space model by one time step.
        """
        self._update_state(
            self._state,
            self._control,
            self._state_space_matrix_A,
            self._state_space_matrix_B,
            np.random.multivariate_normal(
                np.zeros(self._state_dim),
                self._process_noise_covariance_Q,
                size=self._number_of_systems,
            ),
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _update_state_with_control(
        state: NDArray[np.float64],
        control: NDArray[np.float64],
        state_space_matrix_A: NDArray[np.float64],
        state_space_matrix_B: NDArray[np.float64],
        noise: NDArray[np.float64],
    ) -> None:
        for i in range(state.shape[0]):
            state[i, :] = (
                state_space_matrix_A @ state[i, :]
                + state_space_matrix_B @ control[i, :]
                + noise[i, :]
            )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _update_state_without_control(
        state: NDArray[np.float64],
        control: NDArray[np.float64],
        state_space_matrix_A: NDArray[np.float64],
        state_space_matrix_B: NDArray[np.float64],
        noise: NDArray[np.float64],
    ) -> None:
        for i in range(state.shape[0]):
            state[i, :] = state_space_matrix_A @ state[i, :] + noise[i, :]

    @System.observation.getter  # type: ignore
    def observation(self) -> NDArray[np.float64]:
        self._update_observation(
            self._state,
            self._observation,
            self._state_space_matrix_C,
            np.random.multivariate_normal(
                np.zeros(self._observation_dim),
                self._observation_noise_covariance_R,
                size=self._number_of_systems,
            ),
        )
        observation: NDArray = super(
            DiscreteTimeLinearSystem, self.__class__
        ).observation.__get__(self)
        return observation

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _update_observation(
        state: NDArray[np.float64],
        observation: NDArray[np.float64],
        state_space_matrix_C: NDArray[np.float64],
        noise: NDArray[np.float64],
    ) -> None:
        for i in range(state.shape[0]):
            observation[i, :] = state_space_matrix_C @ state[i, :] + noise[i, :]


class MassSpringDamperSystem(DiscreteTimeLinearSystem):
    str_type_observation_choices = Literal["all_positions", "last_position"]

    def __init__(
        self,
        number_of_connections: int,
        mass: float = 1.0,
        spring_constant: float = 1.0,
        damping_coefficient: float = 1.0,
        time_step: float = 0.01,
        observation_choice: str_type_observation_choices = "last_position",
        **kwargs: Any,
    ) -> None:
        assert isPositiveInteger(
            number_of_connections
        ), "number_of_connections should be an integer value that is greater or equal to 1"
        assert observation_choice in get_args(
            self.str_type_observation_choices
        ), f"observation_choice should be one of {get_args(self.str_type_observation_choices)}"
        self.number_of_connections = int(number_of_connections)
        self.discretization_time_step = time_step
        self.mass = mass
        self.spring = spring_constant
        self.damping = damping_coefficient

        matrix_A = np.zeros(
            (2 * self.number_of_connections, 2 * self.number_of_connections)
        )
        matrix_A[: self.number_of_connections, self.number_of_connections :] = (
            np.identity(self.number_of_connections)
        )
        matrix_A[self.number_of_connections, 0] = -self.spring / self.mass
        matrix_A[self.number_of_connections, self.number_of_connections] = (
            -self.damping / self.mass
        )
        for i in range(1, self.number_of_connections):
            position_index = i - 1
            velocity_index = self.number_of_connections + i - 1
            matrix_A[
                velocity_index + 0, position_index : position_index + 2
            ] += (np.array([-1, 1]) * self.spring / self.mass)
            matrix_A[
                velocity_index + 1, position_index : position_index + 2
            ] -= (np.array([-1, 1]) * self.spring / self.mass)
            matrix_A[
                velocity_index + 0, velocity_index : velocity_index + 2
            ] += (np.array([-1, 1]) * self.damping / self.mass)
            matrix_A[
                velocity_index + 1, velocity_index : velocity_index + 2
            ] -= (np.array([-1, 1]) * self.damping / self.mass)

        self.observation_choice = observation_choice
        match self.observation_choice:
            case "all_positions":
                matrix_C = np.zeros(
                    (self.number_of_connections, 2 * self.number_of_connections)
                )
                matrix_C[:, self.number_of_connections :] = np.identity(
                    self.number_of_connections
                )
            case "last_position":
                matrix_C = np.zeros((1, 2 * self.number_of_connections))
                matrix_C[0, -1] = 1

        # TODO: covariance matrices should be calculated based on the time_step parameter

        super().__init__(
            state_space_matrix_A=expm(matrix_A * self.discretization_time_step),
            state_space_matrix_C=matrix_C,
            **kwargs,
        )
