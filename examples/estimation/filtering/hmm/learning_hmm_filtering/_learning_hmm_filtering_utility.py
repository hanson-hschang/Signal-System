from typing import Any, Generator, Optional, Tuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ss.system.markov import one_hot_encoding
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


def get_observation_model(
    transition_probability_matrix: NDArray,
    emission_probability_matrix: NDArray,
    future_time_steps: int = 0,
) -> Any:
    @njit(cache=True)  # type: ignore
    def observation_model(
        estimated_state: NDArray,
        transition_probability_matrix: NDArray = transition_probability_matrix,
        emission_probability_matrix: NDArray = emission_probability_matrix,
        future_time_steps: int = future_time_steps,
    ) -> NDArray:
        for _ in range(future_time_steps):
            estimated_state = estimated_state @ transition_probability_matrix
        estimated_next_observation: NDArray = (
            estimated_state @ emission_probability_matrix
        )
        return estimated_next_observation

    return observation_model


def observation_generator(
    observation_trajectory: NDArray[np.int64],
    discrete_observation_dim: Optional[int] = None,
) -> Generator[Tuple[NDArray, NDArray], None, None]:
    """
    Generate the pair (observation, next_observation) from the observation_trajectory over the time_horizon.

    Parameters
    ----------
    observation_trajectory : NDArray[np.int64]
        shape = (number_of_systems, 1, time_horizon)
    discrete_observation_dim : int, optional
        The dimension of discrete observations.
        If not provided, it will be inferred from the observation_trajectory.

    Yields
    ------
    observation : NDArray
        shape = (number_of_systems, 1)
    next_observation : NDArray
        shape = (number_of_systems, discrete_observation_dim)
        one-hot encoding of the next observation.
    """
    time_horizon = observation_trajectory.shape[-1]
    if discrete_observation_dim is None:
        discrete_observation_dim = int(np.max(observation_trajectory)) + 1
    observation_encoder_basis = np.identity(
        discrete_observation_dim, dtype=np.float64
    )
    for k in range(time_horizon - 1):
        observation = observation_trajectory[..., k]
        next_observation = one_hot_encoding(
            observation_trajectory[:, 0, k + 1],
            observation_encoder_basis,
        )
        yield observation, next_observation


@njit(cache=True)  # type: ignore
def cross_entropy(
    input_probability: NDArray,
    target_probability: NDArray,
) -> float:
    """
    Compute the batch size cross-entropy loss, which is defined as
    :math: `-\\frac{1}{N} \\sum_{i=1}^{N} \\sum_{j=1}^{C} p^*_{ij} \\log(p_{ij})`
    where :math: `N` is the batch size, :math: `C` is the number of classes,
    :math: `p^*_{ij}` is the `target_probability`, and :math: `p_{ij}` is the `input_probability`.

    Parameters
    ----------
    input_probability : NDArray
        probability of input with shape (num_classes,) or (batch_size, num_classes)
        values should be in the range of :math: `(0, 1]`
    target_probability : NDArray
        probability of target with the same shape as input_probability
        values should be in the range of :math: `[0, 1]`

    Returns
    -------
    loss : float
        The cross-entropy loss of the input and target probability.
    """

    num_classes = input_probability.shape[1]
    loss = (
        -np.mean(target_probability * np.log(input_probability)) * num_classes
    )
    return float(loss)
