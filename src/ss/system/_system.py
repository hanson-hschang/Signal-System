from typing import Union

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ss.tool.assertion import is_nonnegative_integer, is_positive_integer
from ss.tool.callback import Callback
from ss.tool.descriptor import MultiSystemTensorDescriptor, ReadOnlyDescriptor


class System:
    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        control_dim: int = 0,
        number_of_systems: int = 1,
    ) -> None:
        assert is_positive_integer(
            state_dim
        ), f"state_dim {state_dim} must be a positive integer"
        assert is_positive_integer(
            observation_dim
        ), f"observation_dim {observation_dim} must be a positive integer"
        assert is_nonnegative_integer(
            control_dim
        ), f"control_dim {control_dim} must be a non-negative integer"
        assert is_positive_integer(
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

    state_dim = ReadOnlyDescriptor[int]()
    observation_dim = ReadOnlyDescriptor[int]()
    control_dim = ReadOnlyDescriptor[int]()
    number_of_systems = ReadOnlyDescriptor[int]()
    state = MultiSystemTensorDescriptor("_number_of_systems", "_state_dim")
    control = MultiSystemTensorDescriptor("_number_of_systems", "_control_dim")

    def create_multiple_systems(self, number_of_systems: int) -> "System":
        """
        Create multiple systems based on the current system.

        Parameters
        ----------
        number_of_systems: int
            The number of systems to be created.

        Returns
        -------
        system: System
            The created multi-system.
        """
        return self.__class__(
            state_dim=self._state_dim,
            observation_dim=self._observation_dim,
            control_dim=self._control_dim,
            number_of_systems=number_of_systems,
        )

    def process(self, time: Union[int, float]) -> Union[int, float]:
        """
        Update the state of each system by one time step based on the current state and control (if existed).

        Parameters
        ----------
        `time: Union[int, float]`
            The current time.

        Returns
        -------
        `time: Union[int, float]`
            The updated time.
        """
        self._update(
            self._state,
            self._compute_state_process(),
            self._compute_process_noise(),
        )
        return time

    def observe(self) -> NDArray[np.float64]:
        """
        Make observation of each system based on the current state.

        Returns
        -------
        `observation: ArrayLike[float]`
            The observation vector of systems. Shape of the array is `(number_of_systems, observation_dim)`.
        """
        self._update(
            self._observation,
            self._compute_observation_process(),
            self._compute_observation_noise(),
        )
        observation: NDArray[np.float64] = (
            self._observation[0]
            if self._number_of_systems == 1
            else self._observation
        )
        return observation

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _update(
        array: NDArray[np.float64],
        process: NDArray[np.float64],
        noise: NDArray[np.float64],
    ) -> None:
        array[:, :] = process + noise

    def _compute_process_noise(self) -> NDArray[np.float64]:
        return np.zeros_like(self._state)

    def _compute_observation_noise(self) -> NDArray[np.float64]:
        return np.zeros_like(self._observation)

    def _compute_state_process(self) -> NDArray[np.float64]:
        return self._state

    def _compute_observation_process(self) -> NDArray[np.float64]:
        return np.zeros_like(self._observation)


class SystemCallback(Callback):
    def __init__(
        self,
        step_skip: int,
        system: System,
    ) -> None:
        assert issubclass(
            type(system), System
        ), f"system must be an instance of System"
        self._system = system
        super().__init__(step_skip)

    def _record(self, time: float) -> None:
        super()._record(time)
        self._callback_params["state"].append(self._system.state.copy())
        self._callback_params["control"].append(self._system.control.copy())
        self._callback_params["observation"].append(
            self._system.observe().copy()
        )
