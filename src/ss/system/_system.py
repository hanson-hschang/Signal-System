from typing import Generic, TypeVar, Union

from pathlib import Path

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ss.utility.assertion import is_nonnegative_integer, is_positive_integer
from ss.utility.callback import Callback
from ss.utility.descriptor import (
    BatchNDArrayDescriptor,
    ReadOnlyDescriptor,
)


class System:
    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        control_dim: int = 0,
        batch_size: int = 1,
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
            batch_size
        ), f"batch_size {batch_size} must be a positive integer"

        self._state_dim = int(state_dim)
        self._observation_dim = int(observation_dim)
        self._control_dim = int(control_dim)
        self._batch_size = int(batch_size)
        self._state = np.zeros((self._batch_size, self._state_dim))
        self._observation = np.zeros((self._batch_size, self._observation_dim))
        self._control = np.zeros((self._batch_size, self._control_dim))

    state_dim = ReadOnlyDescriptor[int]()
    observation_dim = ReadOnlyDescriptor[int]()
    control_dim = ReadOnlyDescriptor[int]()
    batch_size = ReadOnlyDescriptor[int]()
    state = BatchNDArrayDescriptor("_batch_size", "_state_dim")
    observation = BatchNDArrayDescriptor("_batch_size", "_observation_dim")
    control = BatchNDArrayDescriptor("_batch_size", "_control_dim")

    def duplicate(self, batch_size: int) -> "System":
        """
        Create multiple systems based on the current system.

        Parameters
        ----------
        batch_size: int
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
            batch_size=batch_size,
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

    def observe(self) -> NDArray:
        """
        Make observation of each system based on the current state.

        Returns
        -------
        `observation: ArrayLike[float]`
            The observation vector of systems. Shape of the array is `(batch_size, observation_dim)`.
        """
        self._update(
            self._observation,
            self._compute_observation_process(),
            self._compute_observation_noise(),
        )
        # observation: NDArray = (
        #     self._observation[0]
        #     if self._batch_size == 1
        #     else self._observation
        # )
        return self.observation

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _update(
        array: NDArray,
        process: NDArray,
        noise: NDArray,
    ) -> None:
        array[:, :] = process + noise

    def _compute_process_noise(self) -> NDArray:
        return np.zeros_like(self._state)

    def _compute_observation_noise(self) -> NDArray:
        return np.zeros_like(self._observation)

    def _compute_state_process(self) -> NDArray:
        return self._state

    def _compute_observation_process(self) -> NDArray:
        return np.zeros_like(self._observation)


S = TypeVar("S", bound="System")


class SystemCallback(Callback, Generic[S]):
    def __init__(
        self,
        step_skip: int,
        system: S,
    ) -> None:
        assert issubclass(
            type(system), System
        ), f"system must be a subclass of {System.__name__}"
        self._system: S = system
        super().__init__(step_skip)

    def _record(self, time: float) -> None:
        super()._record(time)
        self._callback_params["state"].append(self._system.state.copy())
        self._callback_params["control"].append(self._system.control.copy())
        self._callback_params["observation"].append(
            self._system.observe().copy()
        )

    def save(self, filename: Union[str, Path]) -> None:
        self.add_meta_info(
            state_dim=self._system.state_dim,
            observation_dim=self._system.observation_dim,
            control_dim=self._system.control_dim,
            batch_size=self._system.batch_size,
        )
        super().save(filename)
