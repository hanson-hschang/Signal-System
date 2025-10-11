from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from ss.estimation import Estimator


class DualFilter(Estimator):
    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        history_horizon: int,
        initial_distribution: ArrayLike | None = None,
        estimation_model: Callable | None = None,
        batch_size: int = 1,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            observation_dim=observation_dim,
            history_horizon=history_horizon,
            estimation_model=estimation_model,
            batch_size=batch_size,
        )

        self._initial_distribution = np.array(
            (
                np.ones(self._state_dim) / self._state_dim
                if initial_distribution is None
                else initial_distribution
            ),
            dtype=np.float64,
        )
        assert (self._initial_distribution.ndim == 1) and (
            self._initial_distribution.shape[0] == self._state_dim
        ), (
            f"initial_distribution must be in the "
            f"shape of {(self._state_dim,) = }. "
            f"initial_distribution given has the shape of "
            f"{self._initial_distribution.shape}."
        )

    def reset(self) -> None:
        super().reset()
        self._estimated_state[:, :] = self._initial_distribution[np.newaxis, :]
        # for i in range(self._batch_size):
        #     self._estimated_state[i, :] = self._initial_distribution.copy()
        # self._observation_history[:, :, :] = 0.0

    def duplicate(self, batch_size: int) -> "DualFilter":
        """
        Create multiple filters based on the current filter.

        Parameters
        ----------
        batch_size: int
            The number of systems to be created.

        Returns
        -------
        filter: Filter
            The created multi-filter.
        """
        return self.__class__(
            state_dim=self._state_dim,
            observation_dim=self._observation_dim,
            history_horizon=self._history_horizon,
            initial_distribution=self._initial_distribution,
            estimation_model=self._estimation_model,
            batch_size=batch_size,
        )


class Filter(DualFilter):
    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        initial_distribution: ArrayLike | None = None,
        estimation_model: Callable | None = None,
        batch_size: int = 1,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            observation_dim=observation_dim,
            history_horizon=1,
            initial_distribution=initial_distribution,
            estimation_model=estimation_model,
            batch_size=batch_size,
        )

    def duplicate(self, batch_size: int) -> "Filter":
        """
        Create multiple filters based on the current filter.

        Parameters
        ----------
        batch_size: int
            The number of systems to be created.

        Returns
        -------
        filter: Filter
            The created multi-filter.
        """
        return self.__class__(
            state_dim=self._state_dim,
            observation_dim=self._observation_dim,
            initial_distribution=self._initial_distribution,
            estimation_model=self._estimation_model,
            batch_size=batch_size,
        )
