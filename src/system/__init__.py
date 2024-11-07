from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

from assertion import isNonNegativeInteger, isPositiveInteger
from tool.matrix_descriptor import MatrixDescriptor


class System:
    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        control_dim: int = 0,
        number_of_systems: int = 1,
        **kwargs: Any,
    ) -> None:
        assert isPositiveInteger(
            state_dim
        ), f"state_dim {state_dim} must be a positive integer"
        assert isPositiveInteger(
            observation_dim
        ), f"observation_dim {observation_dim} must be a positive integer"
        assert isNonNegativeInteger(
            control_dim
        ), f"control_dim {control_dim} must be a non-negative integer"
        assert isPositiveInteger(
            number_of_systems
        ), f"number_of_systems {number_of_systems} must be a positive integer"

        self._state_dim = int(state_dim)
        self._observation_dim = int(observation_dim)
        self._control_dim = int(control_dim)
        self._number_of_systems = int(number_of_systems)
        self._state = np.zeros(
            (self._number_of_systems, self._state_dim), dtype=np.float64
        )
        self._observation = np.zeros(
            (self._number_of_systems, self._observation_dim), dtype=np.float64
        )
        self._control = np.zeros(
            (self._number_of_systems, self._control_dim), dtype=np.float64
        )

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

    @property
    def control_dim(self) -> int:
        """
        `control_dim: int`
            The dimension of the control vector.
        """
        return self._control_dim

    @property
    def number_of_systems(self) -> int:
        """
        `number_of_systems: int`
            The number of systems.
        """
        return self._number_of_systems

    state = MatrixDescriptor("number_of_systems", "state_dim")

    @property
    def observation(self) -> NDArray[np.float64]:
        """
        `observation: ArrayLike[float]`
            The observation vector of systems. Shape of the array is `(number_of_systems, observation_dim)`.
        """
        if self._number_of_systems == 1:
            return self._observation.squeeze()
        return self._observation

    control = MatrixDescriptor("number_of_systems", "control_dim")


class DynamicMixin:
    def __init__(
        self,
        time_step: Union[int, float] = 1,
        **kwargs: Any,
    ) -> None:
        assert (
            isinstance(time_step, (int, float)) and time_step > 0
        ), f"time_step {time_step} must be a positive number"
        self._time_step = time_step
        super().__init__(**kwargs)

    @property
    def time_step(self) -> Union[int, float]:
        """
        `time_step: Union[int, float]`
            The time step of the systems.
        """
        return self._time_step

    def update(self) -> None:
        """
        Update the state of each system by one time step.
        """
        pass
