from typing import Callable

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ss.tool.assertion import isPositiveInteger
from ss.tool.descriptor import MultiSystemTensorDescriptor, ReadOnlyDescriptor


class Estimator:
    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        horizon_of_observation_history: int = 1,
        number_of_systems: int = 1,
    ) -> None:
        assert isPositiveInteger(
            state_dim
        ), f"state_dim {state_dim} must be a positive integer"
        assert isPositiveInteger(
            observation_dim
        ), f"observation_dim {observation_dim} must be a positive integer"
        assert isPositiveInteger(
            horizon_of_observation_history
        ), f"horizon_of_observation_history {horizon_of_observation_history} must be a positive integer"
        assert isPositiveInteger(
            number_of_systems
        ), f"number_of_systems {number_of_systems} must be a positive integer"

        self._state_dim = int(state_dim)
        self._observation_dim = int(observation_dim)
        self._horizon_of_observation_history = int(
            horizon_of_observation_history
        )
        self._number_of_systems = int(number_of_systems)
        self._estimated_state = np.zeros(
            (self._number_of_systems, self._state_dim), dtype=np.float64
        )
        self._observation_history = np.zeros(
            (
                self._number_of_systems,
                self._observation_dim,
                self._horizon_of_observation_history,
            ),
            dtype=np.float64,
        )

    state_dim = ReadOnlyDescriptor[int]()
    observation_dim = ReadOnlyDescriptor[int]()
    number_of_observation_history = ReadOnlyDescriptor[int]()
    number_of_systems = ReadOnlyDescriptor[int]()
    estimated_state = MultiSystemTensorDescriptor(
        "_number_of_systems", "_state_dim"
    )
    observation_history = MultiSystemTensorDescriptor(
        "_number_of_systems",
        "_observation_dim",
        "_horizon_of_observation_history",
    )

    def update(self, observation: ArrayLike) -> None:
        observation = np.array(observation, dtype=np.float64)
        if observation.ndim == 1:
            observation = observation[np.newaxis, :]
        assert observation.shape == (
            self._number_of_systems,
            self._observation_dim,
        ), (
            f"argument observation shape {observation.shape} does not match with"
            f"required observation shape {(self._number_of_systems, self._observation_dim)}."
        )
        self._update_observation(
            observation=observation,
            observation_history=self._observation_history,
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _update_observation(
        observation: NDArray[np.float64],
        observation_history: NDArray[np.float64],
    ) -> None:
        observation_history[:, :, -1] = observation
        for i in range(observation_history.shape[0]):
            for j in range(observation_history.shape[1]):
                observation_history[i, j, :] = np.roll(
                    observation_history[i, j, :], 1
                )

    def estimate(self) -> NDArray[np.float64]:
        self._update(
            self._estimated_state,
            self._compute_estimation_process(),
        )
        return self._estimated_state

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _update(
        array: NDArray[np.float64],
        process: NDArray[np.float64],
    ) -> None:
        array[...] = process

    def _compute_estimation_process(self) -> NDArray[np.float64]:
        return np.zeros_like(self._estimated_state)
