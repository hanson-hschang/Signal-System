from typing import Callable, Optional

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ss.utility.assertion.validator import PositiveIntegerValidator
from ss.utility.callback import Callback
from ss.utility.descriptor import (
    BatchNDArrayDescriptor,
    BatchNDArrayReadOnlyDescriptor,
    ReadOnlyDescriptor,
)


class Estimator:
    class _StateDimValidator(PositiveIntegerValidator):
        def __init__(self, state_dim: int) -> None:
            super().__init__(state_dim)

    class _ObservationDimValidator(PositiveIntegerValidator):
        def __init__(self, observation_dim: int) -> None:
            super().__init__(observation_dim)

    class _HorizonOfObservationHistoryValidator(PositiveIntegerValidator):
        def __init__(self, horizon_of_observation_history: int) -> None:
            super().__init__(horizon_of_observation_history)

    class _BatchSizeValidator(PositiveIntegerValidator):
        def __init__(self, batch_size: int) -> None:
            super().__init__(batch_size)

    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        horizon_of_observation_history: int = 1,
        estimation_model: Optional[Callable] = None,
        batch_size: int = 1,
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
        self._batch_size = self._BatchSizeValidator(batch_size).get_value()

        self._estimated_state = np.zeros(
            (self._batch_size, self._state_dim),
            dtype=np.float64,
        )
        self._observation_history = np.zeros(
            (
                self._batch_size,
                self._observation_dim,
                self._horizon_of_observation_history,
            ),
            dtype=np.float64,
        )

        # TODO: The following should be implemented as a validator for estimation_model.
        if estimation_model is None:

            @njit(cache=True)  # type: ignore
            def _estimation_model(
                estimated_state: NDArray[np.float64],
                batch_size: int = self.batch_size,
            ) -> NDArray[np.float64]:
                return np.full((batch_size, 1), np.nan)

            estimation_model = _estimation_model
        self._estimation_model = estimation_model

        self._estimation: NDArray[np.float64] = self._estimation_model(
            self._estimated_state
        )
        self._estimation_dim = self._estimation.shape[1]

    state_dim = ReadOnlyDescriptor[int]()
    observation_dim = ReadOnlyDescriptor[int]()
    estimation_dim = ReadOnlyDescriptor[int]()
    number_of_observation_history = ReadOnlyDescriptor[int]()
    batch_size = ReadOnlyDescriptor[int]()
    estimated_state = BatchNDArrayDescriptor(
        "_batch_size",
        "_state_dim",
    )
    observation_history = BatchNDArrayReadOnlyDescriptor(
        "_batch_size",
        "_observation_dim",
        "_horizon_of_observation_history",
    )
    estimation = BatchNDArrayReadOnlyDescriptor(
        "_batch_size",
        "_estimation_dim",
    )

    def duplicate(self, batch_size: int) -> "Estimator":
        """
        Create multiple estimators based on the current estimator.

        Parameters
        ----------
        batch_size: int
            The number of systems to be created.

        Returns
        -------
        estimator: Estimator
            The created multi-estimator.
        """
        return self.__class__(
            state_dim=self._state_dim,
            observation_dim=self._observation_dim,
            horizon_of_observation_history=self._horizon_of_observation_history,
            estimation_model=self._estimation_model,
            batch_size=batch_size,
        )

    def update(self, observation: ArrayLike) -> None:
        """
        Update the observation history with the given observation.

        Parameters
        ----------
        observation : ArrayLike
            shape = (batch_size, observation_dim)
            The observation to be updated.
        """
        observation = np.array(observation, dtype=np.float64)
        if observation.ndim == 1:
            observation = observation[np.newaxis, :]
        assert observation.shape == (
            self._batch_size,
            self._observation_dim,
        ), (
            f"observation must be in the shape of {(self._batch_size, self._observation_dim) = }. "
            f"observation given has the shape of {observation.shape}."
        )
        self._update_observation(
            observation=observation,
            observation_history=self._observation_history,
        )
        self._update(
            self._estimated_state,
            self._compute_estimated_state_process(),
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _update_observation(
        observation: NDArray[np.float64],
        observation_history: NDArray[np.float64],
    ) -> None:
        batch_size, observation_dim, _ = observation_history.shape
        observation_history[:, :, -1] = observation
        for i in range(batch_size):
            for m in range(observation_dim):
                observation_history[i, m, :] = np.roll(
                    observation_history[i, m, :], 1
                )

    def estimate(self) -> NDArray:
        self._update(
            self._estimation,
            self._estimation_model(self._estimated_state),
        )
        return self.estimation

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _update(
        array: NDArray[np.float64],
        process: NDArray[np.float64],
    ) -> None:
        array[...] = process

    def _compute_estimated_state_process(self) -> NDArray[np.float64]:
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
        self._callback_params["estimation"].append(
            self._estimator.estimation.copy()
        )
