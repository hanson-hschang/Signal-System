from typing import Callable, Optional

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ss.estimation import EstimatorCallback
from ss.estimation.filtering import Filter
from ss.system.markov import HiddenMarkovModel


class HmmFilter(Filter):
    def __init__(
        self,
        system: HiddenMarkovModel,
        initial_distribution: Optional[ArrayLike] = None,
        estimation_model: Optional[Callable] = None,
        number_of_systems: Optional[int] = None,
    ) -> None:
        assert issubclass(type(system), HiddenMarkovModel), (
            f"system must be an instance of HiddenMarkovModel or its subclasses. "
            f"system given is an instance of {type(system)}."
        )
        self._system = system
        number_of_systems = (
            system.number_of_systems
            if number_of_systems is None
            else number_of_systems
        )
        super().__init__(
            state_dim=self._system.discrete_state_dim,
            observation_dim=self._system.observation_dim,
            initial_distribution=initial_distribution,
            estimation_model=estimation_model,
            number_of_systems=number_of_systems,
        )
        self.reset()

    def duplicate(self, number_of_systems: int) -> "HmmFilter":
        """
        Create multiple filters based on the current filter.

        Parameters
        ----------
        number_of_systems: int
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
            number_of_systems=number_of_systems,
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
        number_of_systems: int = estimated_state.shape[0]

        # prediction step based on model process (predicted probability)
        estimated_state[:, :] = estimated_state @ transition_matrix

        # update step based on observation (unnormalized conditional probability)
        # the transpose operation is for the purpose of the multi-system case
        estimated_state[:, :] = (
            estimated_state * emission_matrix[:, observation].T
        )

        # normalization step (conditional probability)
        for i in range(number_of_systems):
            estimated_state[i, :] /= np.sum(estimated_state[i, :])

        return estimated_state


class HmmFilterCallback(EstimatorCallback):
    def __init__(
        self,
        step_skip: int,
        estimator: HmmFilter,
    ) -> None:
        assert issubclass(type(estimator), HmmFilter), (
            f"estimator must be an instance of HmmFilter or its subclasses. "
            f"estimator given is an instance of {type(estimator)}."
        )
        super().__init__(step_skip, estimator)
        self._estimator: HmmFilter = estimator
