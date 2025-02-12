from typing import Any, Generator, Optional, Tuple

from dataclasses import dataclass

import numpy as np
import torch
from numba import njit
from numpy.typing import NDArray

from ss.estimation.filtering.hmm import HmmFilter
from ss.estimation.filtering.hmm.learning import module as Module
from ss.system.markov import one_hot_encoding
from ss.utility.learning.mode import LearningMode
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


@njit(cache=True)  # type: ignore
def cross_entropy(
    input_probability: NDArray[np.float64],
    target_probability: NDArray[np.float64],
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
    if input_probability.ndim == 1:
        loss = -np.sum(target_probability * np.log(input_probability))
    else:
        num_classes = input_probability.shape[1]
        loss = (
            -np.mean(target_probability * np.log(input_probability))
            * num_classes
        )
    return float(loss)


def compute_optimal_loss(
    filter: HmmFilter,
    observation_trajectory: NDArray,
) -> float:
    """
    Compute the empirical optimal loss of the hmm-filter.

    Parameters
    ----------
    filter : HiddenMarkovModelFilter
        The filter to be used for the estimation.
    observation_trajectory : NDArray
        shape = (number_of_systems, 1, time_horizon)

    Returns
    -------
    average_loss : float
        The average loss of the optimal estimation.
    """
    time_horizon = observation_trajectory.shape[-1]
    loss_trajectory = np.empty(time_horizon - 1)
    for k, (observation, next_observation_one_hot) in logger.progress_bar(
        enumerate(
            observation_generator(
                observation_trajectory=observation_trajectory,
                discrete_observation_dim=filter.estimation_dim,
            )
        ),
        total=time_horizon - 1,
    ):
        filter.update(observation=observation)
        filter.estimate()
        loss_trajectory[k] = cross_entropy(
            input_probability=filter.estimation,
            target_probability=next_observation_one_hot,
        )
    average_loss = float(np.mean(loss_trajectory))
    return average_loss


def compute_layer_loss_trajectory(
    learning_filter: Module.LearningHmmFilter,
    observation_trajectory: NDArray,
) -> Tuple[NDArray, NDArray]:
    """
    Compute the loss of the learning_filter over the observation_trajectory.

    Parameters
    ----------
    learning_filter : LearningHiddenMarkovModelFilter
        The learning filter to be used for the estimation.
    observation_trajectory : NDArray
        shape = (number_of_systems, 1, time_horizon)

    Returns
    -------
    loss_trajectory : NDArray
        The loss trajectory of the learning_filter for each layer.
        shape = (number_of_systems, layer_dim, time_horizon - 1)
    average_loss : NDArray
        The average loss of the learning_filter for each layer.
        shape = (layer_dim,)
    """
    if observation_trajectory.ndim == 2:
        observation_trajectory = observation_trajectory[np.newaxis, ...]
    number_of_systems, _, time_horizon = observation_trajectory.shape
    layer_dim = learning_filter.layer_dim + 1
    loss_trajectory = np.empty(
        (number_of_systems, layer_dim, time_horizon - 1)
    )
    with LearningMode.inference(learning_filter):
        for k, (observation, next_observation) in logger.progress_bar(
            enumerate(
                observation_generator(
                    observation_trajectory=observation_trajectory,
                    discrete_observation_dim=learning_filter.discrete_observation_dim,
                )
            ),
            total=time_horizon - 1,
        ):
            learning_filter.update(torch.tensor(observation))
            learning_filter.estimate()
            layer_output = learning_filter.estimation.numpy()
            for l in range(layer_dim):
                loss_trajectory[:, l, k] = cross_entropy(
                    input_probability=(
                        layer_output[:, l, :]
                        if number_of_systems > 1
                        else layer_output[l]
                    ),
                    target_probability=next_observation,
                )
    average_loss = np.mean(loss_trajectory, axis=(0, 2))

    return loss_trajectory, average_loss


@dataclass
class FilterResultTrajectory:
    loss: NDArray
    estimated_next_observation_probability: NDArray


def compute_loss_trajectory(
    filter: HmmFilter,
    learning_filter: Module.LearningHmmFilter,
    observation_trajectory: NDArray,
) -> Tuple[FilterResultTrajectory, FilterResultTrajectory]:
    """
    Compute the loss trajectory of the filter and learning_filter.

    Parameters
    ----------
    filter : HiddenMarkovModelFilter
        The filter to be used for the estimation.
    learning_filter : LearningHiddenMarkovModelFilter
        The learning filter to be used for the estimation.
    observation_trajectory : NDArray
        shape = (1, time_horizon)

    Returns
    -------
    filter_result_trajectory : FilterResultTrajectory
        The result of the filter estimation, including the loss and estimated_next_observation_probability.
    learning_filter_result_trajectory : FilterResultTrajectory
        The result of the learning_filter, including the loss and estimated_next_observation_probability.
    """
    time_horizon = observation_trajectory.shape[-1]
    discrete_observation_dim = filter.estimation_dim
    filter_loss_trajectory = np.empty(time_horizon - 1)
    learning_filter_loss_trajectory = np.empty(time_horizon - 1)
    filter_estimated_next_observation_probability = np.empty(
        (discrete_observation_dim, time_horizon - 1)
    )
    learning_filter_estimated_next_observation_probability = np.empty(
        (discrete_observation_dim, time_horizon - 1)
    )
    with LearningMode.inference(learning_filter):
        for k, (observation, next_observation) in logger.progress_bar(
            enumerate(
                observation_generator(
                    observation_trajectory=observation_trajectory,
                    discrete_observation_dim=filter.estimation_dim,
                )
            ),
            total=time_horizon - 1,
        ):
            filter.update(observation)
            filter.estimate()
            learning_filter.update(torch.tensor(observation))
            learning_filter.estimate()
            filter_estimated_next_observation_probability[:, k] = (
                filter.estimation
            )
            filter_loss_trajectory[k] = cross_entropy(
                input_probability=filter.estimation,
                target_probability=next_observation,
            )
            learning_filter_estimated_next_observation_probability[:, k] = (
                learning_filter.predicted_next_observation_probability.numpy()
            )
            learning_filter_loss_trajectory[k] = cross_entropy(
                input_probability=learning_filter.predicted_next_observation_probability.numpy(),
                target_probability=next_observation,
            )

    filter_mean_loss_trajectory = np.empty(time_horizon - 1)
    learning_filter_mean_loss_trajectory = np.empty(time_horizon - 1)
    for k in range(time_horizon - 1):
        filter_mean_loss_trajectory[k] = np.sum(filter_loss_trajectory[:k]) / (
            k + 1
        )
        learning_filter_mean_loss_trajectory[k] = np.sum(
            learning_filter_loss_trajectory[:k]
        ) / (k + 1)

    filter_result_trajectory = FilterResultTrajectory(
        loss=filter_mean_loss_trajectory,
        estimated_next_observation_probability=filter_estimated_next_observation_probability,
    )
    learning_filter_result_trajectory = FilterResultTrajectory(
        loss=learning_filter_mean_loss_trajectory,
        estimated_next_observation_probability=learning_filter_estimated_next_observation_probability,
    )
    return filter_result_trajectory, learning_filter_result_trajectory
