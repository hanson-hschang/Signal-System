from typing import Any, Generator, Optional, Tuple

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from numba import njit
from numpy.typing import NDArray

from ss.estimation.filtering.hmm import HmmFilter
from ss.estimation.filtering.hmm.learning.module import LearningHmmFilter
from ss.estimation.filtering.hmm.learning.module.config import EstimationConfig
from ss.system.markov import one_hot_encoding
from ss.utility.learning.process import BaseLearningProcess
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


def cross_entropy(
    input_probability: NDArray[np.float64],
    target_probability: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute the batch size cross-entropy loss, which is defined as
    :math: `-\\sum_{c=1}^{C} p^*_{nc} \\log(p_{nc})`
    where :math: `C` is the number of classes, :math: `n \\in N` with `N` as the batch size,
    :math: `p^*_{nc}` is the `target_probability`, and :math: `p_{nc}` is the `input_probability`.

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
    loss_mean : float
        The average loss of the optimal estimation.
    """
    number_of_systems, _, time_horizon = observation_trajectory.shape
    loss_trajectory = np.empty((number_of_systems, time_horizon - 1))
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
        loss_trajectory[:, k] = cross_entropy(
            input_probability=filter.estimation,
            target_probability=next_observation_one_hot,
        )
    loss_mean = float(loss_trajectory.mean())
    return loss_mean


def compute_layer_loss_trajectory(
    learning_filter: LearningHmmFilter,
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
    loss_mean_over_layers : NDArray
        The average loss of the learning_filter for each layer.
        shape = (layer_dim,)
    """
    if observation_trajectory.ndim == 2:
        observation_trajectory = observation_trajectory[np.newaxis, ...]
    number_of_systems, _, time_horizon = observation_trajectory.shape
    layer_dim = learning_filter.layer_dim
    loss_trajectory = np.empty(
        (number_of_systems, layer_dim, time_horizon - 1)
    )
    learning_filter.estimation_option = (
        EstimationConfig.Option.PREDICTED_NEXT_OBSERVATION_PROBABILITY_OVER_LAYERS
    )

    with BaseLearningProcess.inference_mode(learning_filter):
        # observation_queue = FixLengthObservationQueue(
        #     max_length=128,
        # )
        learning_filter.reset()
        for k, (observation, next_observation) in logger.progress_bar(
            enumerate(
                observation_generator(
                    observation_trajectory=observation_trajectory,
                    discrete_observation_dim=learning_filter.discrete_observation_dim,
                )
            ),
            total=time_horizon - 1,
        ):
            # learning_filter.reset()
            # observation_queue.append(observation)
            # learning_filter.update(torch.tensor(observation_queue.to_numpy()))

            learning_filter.update(torch.tensor(observation))
            learning_filter.estimate()
            layer_output = learning_filter.estimation.numpy()
            # predicted_next_observation_probability = (
            #     learning_filter.predicted_next_observation_probability.numpy()
            # )
            # for i in range(2048):
            #     assert np.allclose(
            #         layer_output[i, 1, :],
            #         predicted_next_observation_probability[i]
            #     ), (layer_output[i, 1, :], predicted_next_observation_probability[i])
            for l in range(layer_dim):
                loss_trajectory[:, l, k] = cross_entropy(
                    input_probability=layer_output[:, l, :],
                    target_probability=next_observation,
                )
    loss_mean_over_layers = np.mean(loss_trajectory, axis=(0, 2))

    return loss_trajectory, loss_mean_over_layers


@dataclass
class FilterResultTrajectory:
    loss: NDArray
    estimated_next_observation_probability: NDArray


def compute_loss_trajectory(
    filter: HmmFilter,
    learning_filter: LearningHmmFilter,
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
    learning_filter.estimation_option = (
        EstimationConfig.Option.PREDICTED_NEXT_OBSERVATION_PROBABILITY_OVER_LAYERS
    )

    with BaseLearningProcess.inference_mode(learning_filter):
        # observation_queue = FixLengthObservationQueue(
        #     max_length=128,
        # )

        learning_filter.reset()
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

            # learning_filter.reset()
            # observation_queue.append(observation[np.newaxis, :])
            # learning_filter.update(torch.tensor(observation_queue.to_numpy()))

            learning_filter.update(torch.tensor(observation))
            learning_filter.estimate()

            filter_estimated_next_observation_probability[:, k] = (
                filter.estimation
            )
            filter_loss_trajectory[k] = cross_entropy(
                input_probability=filter.estimation[np.newaxis, :],
                target_probability=next_observation[np.newaxis, :],
            )[0]

            learning_filter_estimated_next_observation_probability[:, k] = (
                learning_filter.predicted_next_observation_probability.numpy()
            )
            learning_filter_loss_trajectory[k] = cross_entropy(
                input_probability=learning_filter.predicted_next_observation_probability.numpy()[
                    np.newaxis, :
                ],
                target_probability=next_observation[np.newaxis, :],
            )[0]

    filter_loss_mean_trajectory = np.empty(time_horizon - 1)
    learning_filter_loss_mean_trajectory = np.empty(time_horizon - 1)
    for k in range(time_horizon - 1):
        filter_loss_mean_trajectory[k] = filter_loss_trajectory[: k + 1].mean()
        learning_filter_loss_mean_trajectory[k] = (
            learning_filter_loss_trajectory[: k + 1].mean()
        )

    filter_result_trajectory = FilterResultTrajectory(
        loss=filter_loss_mean_trajectory,
        estimated_next_observation_probability=filter_estimated_next_observation_probability,
    )
    learning_filter_result_trajectory = FilterResultTrajectory(
        loss=learning_filter_loss_mean_trajectory,
        estimated_next_observation_probability=learning_filter_estimated_next_observation_probability,
    )
    return filter_result_trajectory, learning_filter_result_trajectory
