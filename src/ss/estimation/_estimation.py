import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ss.utility.assertion.validator import PositiveIntegerValidator
from ss.utility.callback import Callback
from ss.utility.descriptor import (
    MultiSystemNDArrayDescriptor,
    MultiSystemNDArrayReadOnlyDescriptor,
    ReadOnlyDescriptor,
)


class Estimator:
    class _StateDimValidator(PositiveIntegerValidator):
        def __init__(self, state_dim: int) -> None:
            super().__init__(state_dim, "state_dim")

    class _ObservationDimValidator(PositiveIntegerValidator):
        def __init__(self, observation_dim: int) -> None:
            super().__init__(observation_dim, "observation_dim")

    class _HorizonOfObservationHistoryValidator(PositiveIntegerValidator):
        def __init__(self, horizon_of_observation_history: int) -> None:
            super().__init__(
                horizon_of_observation_history, "horizon_of_observation_history"
            )

    class _NumberOfSystemsValidator(PositiveIntegerValidator):
        def __init__(self, number_of_systems: int) -> None:
            super().__init__(number_of_systems, "number_of_systems")

    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        horizon_of_observation_history: int = 1,
        number_of_systems: int = 1,
    ) -> None:
        self._state_dim = self._StateDimValidator(state_dim).get_value()
        self._observation_dim = self._ObservationDimValidator(
            observation_dim
        ).get_value()
        self._horizon_of_observation_history = (
            self._HorizonOfObservationHistoryValidator(
                horizon_of_observation_history
            ).get_value()
        )
        self._number_of_systems = self._NumberOfSystemsValidator(
            number_of_systems
        ).get_value()

        self._estimated_state = np.zeros(
            (self._number_of_systems, self._state_dim),
            dtype=np.float64,
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
    estimated_state = MultiSystemNDArrayDescriptor(
        "_number_of_systems",
        "_state_dim",
    )
    observation_history = MultiSystemNDArrayReadOnlyDescriptor(
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
            f"observation must be in the shape of {(self._number_of_systems, self._observation_dim) = }. "
            f"observation given has the shape of {observation.shape}."
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
        number_of_systems, observation_dim, _ = observation_history.shape
        observation_history[:, :, -1] = observation
        for i in range(number_of_systems):
            for m in range(observation_dim):
                observation_history[i, m, :] = np.roll(
                    observation_history[i, m, :], 1
                )

    def estimate(self) -> None:
        self._update(
            self._estimated_state,
            self._compute_estimation_process(),
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _update(
        array: NDArray[np.float64],
        process: NDArray[np.float64],
    ) -> None:
        array[...] = process

    def _compute_estimation_process(self) -> NDArray[np.float64]:
        return np.zeros_like(self._estimated_state)


class EstimatorCallback(Callback):
    def __init__(
        self,
        step_skip: int,
        estimator: Estimator,
    ) -> None:
        assert issubclass(type(estimator), Estimator), (
            f"estimator must be an instance of Estimator or its subclasses. "
            f"estimator given is an instance of {type(estimator)}."
        )
        self._estimator = estimator
        super().__init__(step_skip)

    def _record(self, time: float) -> None:
        super()._record(time)
        self._callback_params["estimated_state"].append(
            self._estimator.estimated_state.copy()
        )
