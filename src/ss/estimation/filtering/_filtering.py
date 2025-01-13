from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike

from ss.estimation import Estimator


class Filter(Estimator):
    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        initial_distribution: Optional[ArrayLike] = None,
        estimation_model: Optional[Callable] = None,
        number_of_systems: int = 1,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            observation_dim=observation_dim,
            horizon_of_observation_history=1,
            estimation_model=estimation_model,
            number_of_systems=number_of_systems,
        )
        if initial_distribution is None:
            initial_distribution = np.ones(self._state_dim) / self._state_dim
        self._initial_distribution = np.array(
            initial_distribution, dtype=np.float64
        )
        assert (self._initial_distribution.ndim == 1) and (
            self._initial_distribution.shape[0] == self._state_dim
        ), (
            f"initial_distribution must be in the shape of {(self._state_dim,) = }. "
            f"initial_distribution given has the shape of {self._initial_distribution.shape}."
        )

    def duplicate(self, number_of_systems: int) -> "Filter":
        """
        Create multiple filters based on the current filter.

        Parameters
        ----------
        number_of_systems: int
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
            number_of_systems=number_of_systems,
        )

    def reset(self) -> None:
        for i in range(self._number_of_systems):
            self._estimated_state[i, :] = self._initial_distribution.copy()
        self._observation_history[:, :, :] = 0.0
