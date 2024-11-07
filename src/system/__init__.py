from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from assertion import isPositiveInteger


class System:
    def __init__(self, state_dim: int, observation_dim: int) -> None:
        super().__init__()
        assert isPositiveInteger(
            state_dim
        ), f"state_dim {state_dim} must be a positive integer"
        assert isPositiveInteger(
            observation_dim
        ), f"observation_dim {observation_dim} must be a positive integer"
        self._state_dim = int(state_dim)
        self._observation_dim = int(observation_dim)
        self._state = np.zeros(self._state_dim, dtype=np.float64)
        self._observation = np.zeros(self._observation_dim, dtype=np.float64)

    @property
    def state(self) -> NDArray[np.float64]:
        """
        `state: ArrayLike[float]`
            The state vector of the system. Length of the vector is `state_dim`.
        """
        return self._state

    @state.setter
    def state(self, state: ArrayLike) -> None:
        state = np.array(state, dtype=np.float64)
        assert state.shape == (
            self._state_dim,
        ), f"state shape {state.shape} must be equal to ({self._state_dim},)."
        self._state = state

    @property
    def observation(self) -> NDArray[np.float64]:
        """
        `observation: ArrayLike[float]`
            The observation vector of the system. Length of the vector is `observation_dim`.
        """
        return self._observation

    @property
    def state_dim(self) -> int:
        """
        `state_dim: int`
            The dimension of the state vector.
        """
        return self._state_dim

    @property
    def observation_dim(self) -> int:
        """
        `observation_dim: int`
            The dimension of the observation vector.
        """
        return self._observation_dim


class DynamicSystem(System):
    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        time_step: Union[int, float] = 1,
    ) -> None:
        super().__init__(state_dim, observation_dim)
        assert (
            isinstance(time_step, (int, float)) and time_step > 0
        ), f"time_step {time_step} must be a positive number"
        self._time_step = time_step
        self.update_observation()

    def update(self) -> None:
        """
        Update the system by one time step.
        """
        self.update_state()
        self.update_observation()

    def update_state(self) -> None:
        """
        Update the state of the system.
        """
        pass

    def update_observation(self) -> None:
        """
        Update the observation of the system.
        """
        pass

    @System.state.setter  # type: ignore
    def state(self, state: ArrayLike) -> None:
        super(DynamicSystem, self.__class__).state.__set__(self, state)  # type: ignore
        self.update_observation()

    @property
    def time_step(self) -> Union[int, float]:
        """
        `time_step: Union[int, float]`
            The time step of the system.
        """
        return self._time_step
