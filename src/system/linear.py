from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from system import DynamicSystem


class DiscreteTimeLinearSystem(DynamicSystem):
    def __init__(
        self,
        state_space_matrix_A: ArrayLike,
        state_space_matrix_C: ArrayLike,
        process_noise_covariance_Q: Optional[ArrayLike] = None,
        observation_noise_covariance_R: Optional[ArrayLike] = None,
    ):
        state_space_matrix_A = np.array(state_space_matrix_A, dtype=np.float64)
        state_space_matrix_C = np.array(state_space_matrix_C, dtype=np.float64)

        assert (
            (len(state_space_matrix_A.shape) == 2) and (state_space_matrix_A.shape[0] == state_space_matrix_A.shape[1])
        ), "state_space_matrix_A should be a square matrix"
        assert (
            len(state_space_matrix_C.shape) == 2
        ), "state_space_matrix_C should be a 2D matrix"
        assert (
            state_space_matrix_A.shape[1] == state_space_matrix_C.shape[1]
        ), "state_space_matrix_A and state_space_matrix_C should have compatible shapes"

        if process_noise_covariance_Q is not None:
            process_noise_covariance_Q = np.array(process_noise_covariance_Q, dtype=np.float64)
            assert (
                (len(process_noise_covariance_Q.shape) == 2) and (process_noise_covariance_Q.shape[0] == process_noise_covariance_Q.shape[1])
            ), "process_noise_covariance_Q should be a square matrix"
            assert (
                process_noise_covariance_Q.shape[0]
                == state_space_matrix_A.shape[0]
            ), "process_noise_covariance_Q and state_space_matrix_A should have compatible shapes"
        else:
            process_noise_covariance_Q = np.zeros(state_space_matrix_A.shape)
        if observation_noise_covariance_R is not None:
            observation_noise_covariance_R = np.array(observation_noise_covariance_R, dtype=np.float64)
            assert (
                (len(observation_noise_covariance_R.shape) == 2) and (observation_noise_covariance_R.shape[0] == observation_noise_covariance_R.shape[1])
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
        self._state_space_matrix_C = state_space_matrix_C
        self._process_noise_covariance_Q = process_noise_covariance_Q
        self._observation_noise_covariance_R = observation_noise_covariance_R

        super().__init__(state_space_matrix_A.shape[0], state_space_matrix_C.shape[0])
    
    @property
    def state_space_matrix_A(self) -> NDArray[np.float64]:
        """
        `state_space_matrix_A: ArrayLike[float]`
            The state space matrix of the system. Shape of the matrix is `(state_dim, state_dim)`.
        """
        return self._state_space_matrix_A
    
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
    
    def update_state(self) -> None:
        """
        Update the state vector using the state space model.
        """
        self._state[:] = self._state_space_matrix_A @ self._state + np.random.multivariate_normal(np.zeros(self._state_dim), self._process_noise_covariance_Q)

    def update_observation(self) -> None:
        """
        Update the observation vector using the state space model.
        """
        self._observation[:] = self._state_space_matrix_C @ self._state + np.random.multivariate_normal(np.zeros(self._observation_dim), self._observation_noise_covariance_R)

def main():
    matrix_A = np.array([[0.9, 0.1], [0.0, 1.0]])
    matrix_C = np.array([[1.0, 0.0]])
    linear_system = DiscreteTimeLinearSystem(matrix_A, matrix_C)
    linear_system.state = np.array([0.0, 9.0])
    print("initial_state = ", linear_system.state)
    for k in range(10):
        linear_system.update()
        print("\n time step = ", k)
        print("state = ", linear_system.state)
        print("observation = ", linear_system.observation)


if __name__ == "__main__":
    main()


    