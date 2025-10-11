from collections import deque
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ss.system.markov import one_hot_encoding
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


def get_estimation_model(
    transition_matrix: NDArray,
    emission_matrix: NDArray,
    future_time_steps: int = 0,
) -> Any:
    @njit(cache=True)  # type: ignore
    def estimation_model(
        estimated_state: NDArray,
        transition_matrix: NDArray = transition_matrix,
        emission_matrix: NDArray = emission_matrix,
        future_time_steps: int = future_time_steps,
    ) -> NDArray:
        for _ in range(future_time_steps):
            estimated_state = estimated_state @ transition_matrix
        estimation: NDArray = estimated_state @ emission_matrix
        return estimation

    return estimation_model


def observation_generator(
    observation_trajectory: NDArray[np.int64],
    discrete_observation_dim: int | None = None,
) -> Generator[tuple[NDArray, NDArray]]:
    """
    Generate the pair (observation, next_observation)
    from the observation_trajectory over the time_horizon.

    Parameters
    ----------
    observation_trajectory : NDArray[np.int64]
        shape = (batch_size, 1, time_horizon)
    discrete_observation_dim : int, optional
        The dimension of discrete observations.
        If not provided, it will be inferred from the observation_trajectory.

    Yields
    ------
    observation : NDArray
        shape = (batch_size, 1)
    next_observation : NDArray
        shape = (batch_size, discrete_observation_dim)
        one-hot encoding of the next observation.
    """
    time_horizon = observation_trajectory.shape[-1]
    if discrete_observation_dim is None:
        discrete_observation_dim = int(np.max(observation_trajectory)) + 1
    observation_encoder_basis = np.identity(
        discrete_observation_dim, dtype=np.float64
    )
    system_dim = True
    if observation_trajectory.ndim == 2:
        observation_trajectory = observation_trajectory[np.newaxis, ...]
        system_dim = False
    for k in range(time_horizon - 1):
        observation = observation_trajectory[..., k]
        next_observation = one_hot_encoding(
            observation_trajectory[:, 0, k + 1],
            observation_encoder_basis,
        )
        if system_dim:
            yield observation, next_observation
        else:
            yield observation[0], next_observation[0]


def cross_entropy(
    input_probability: NDArray[np.float64],
    target_probability: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute the batch size cross-entropy loss, which is defined as
    :math: `-\\sum_{c=1}^{C} p^*_{nc} \\log(p_{nc})`
    where :math: `C` is the number of classes,
    :math: `n \\in N` with `N` as the batch size,
    :math: `p^*_{nc}` is the `target_probability`,
    and :math: `p_{nc}` is the `input_probability`.

    Parameters
    ----------
    input_probability : NDArray
        probability of input with shape (batch_size, num_classes)
        values should be in the range of :math: `(0, 1]`
    target_probability : NDArray
        probability of target with the same shape as input_probability
        values should be in the range of :math: `[0, 1]`

    Returns
    -------
    losses : NDArray
        losses with shape (batch_size,)
        The batch size cross-entropy loss of the input and target probability.
    """
    losses = -np.sum(
        target_probability * np.log(input_probability),
        axis=1,
    )
    return np.array(losses)


class FixLengthObservationQueue(deque):
    def __init__(self, max_length: int) -> None:
        super().__init__(maxlen=max_length)

    def append(self, observation: NDArray) -> None:
        super().append(observation[:, 0])

    def to_numpy(self) -> NDArray:
        return np.array(self).T


@dataclass
class FilterResultTrajectory:
    loss: NDArray
    estimated_next_observation_probability: NDArray
