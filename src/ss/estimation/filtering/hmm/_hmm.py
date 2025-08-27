from typing import Callable, Optional

import numpy as np
from numba import njit, prange
from numpy.typing import ArrayLike, NDArray

from ss.estimation import EstimatorCallback
from ss.estimation.filtering import Filter
from ss.system.markov import HiddenMarkovModel
from ss.utility.descriptor import ReadOnlyDescriptor


class HmmFilter(Filter):
    def __init__(
        self,
        system: HiddenMarkovModel,
        initial_distribution: Optional[ArrayLike] = None,
        estimation_model: Optional[Callable] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        assert issubclass(type(system), HiddenMarkovModel), (
            f"system must be an instance of HiddenMarkovModel or its subclasses. "
            f"system given is an instance of {type(system)}."
        )
        self._system = system
        batch_size = system.batch_size if batch_size is None else batch_size
        super().__init__(
            state_dim=self._system.discrete_state_dim,
            observation_dim=self._system.observation_dim,
            initial_distribution=initial_distribution,
            estimation_model=estimation_model,
            batch_size=batch_size,
        )
        self._discrete_observation_dim = self._system.discrete_observation_dim
        self.reset()

    discrete_observation_dim = ReadOnlyDescriptor[int]()

    def duplicate(self, batch_size: int) -> "HmmFilter":
        """
        Create multiple filters based on the current filter.

        Parameters
        ----------
        batch_size: int
            The number of systems to be created.

        Returns
        -------
        filter: HmmFilter
            The created multi-filter.
        """
        return self.__class__(
            system=self._system,
            initial_distribution=self._initial_distribution,
            estimation_model=self._estimation_model,
            batch_size=batch_size,
        )

    def _compute_estimated_state_process(self) -> NDArray[np.float64]:
        estimated_state: NDArray[np.float64] = self._estimated_state_process(
            estimated_state=self._estimated_state,
            observation=self._observation_history[:, 0, 0].astype(np.int64),
            transition_matrix=self._system.transition_matrix,
            emission_matrix=self._system.emission_matrix,
        )
        return estimated_state

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _estimated_state_process(
        estimated_state: NDArray[np.float64],
        observation: NDArray[np.int64],
        transition_matrix: NDArray[np.float64],
        emission_matrix: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        batch_size: int = estimated_state.shape[0]

        # update step based on observation (unnormalized conditional probability)
        # the transpose operation is for the purpose of the multi-system case
        estimated_state[:, :] = (
            estimated_state * emission_matrix[:, observation].T
        )

        # normalization step (conditional probability)
        for i in prange(batch_size):
            estimated_state[i, :] /= np.sum(estimated_state[i, :])

        # prediction step based on model process (predicted probability)
        estimated_state[:, :] = estimated_state @ transition_matrix

        return estimated_state


class HmmFilterCallback(EstimatorCallback):
    def __init__(
        self,
        step_skip: int,
        filter: HmmFilter,
    ) -> None:
        assert issubclass(type(filter), HmmFilter), (
            f"filter must be an instance of HmmFilter or its subclasses. "
            f"filter given is an instance of {type(filter)}."
        )
        super().__init__(step_skip, filter)
        self._estimator: HmmFilter = filter
