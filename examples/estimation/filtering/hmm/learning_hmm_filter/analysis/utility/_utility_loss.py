from typing import Generic, Protocol, Tuple, TypeVar, Union

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from ss.estimation.filtering.hmm import HmmFilter
from ss.estimation.filtering.hmm.learning.module import LearningHmmFilter
from ss.utility.descriptor import ReadOnlyDescriptor
from ss.utility.learning.process import BaseLearningProcess
from ss.utility.logging import Logging

from ._utility import (
    FilterResultTrajectory,
    FixLengthObservationQueue,
    cross_entropy,
    observation_generator,
)

logger = Logging.get_logger(__name__)

L = TypeVar("L", float, NDArray)


class LossConversion:
    def __init__(self, log_base: float = np.e) -> None:
        self._log_base = log_base
        self._scaling: float = 1.0 / np.log(log_base)

    @property
    def log_base(self) -> float:
        return self._log_base

    @property
    def scaling(self) -> float:
        return self._scaling

    def __call__(self, loss: L) -> L:
        return loss * self._scaling


T = TypeVar("T", bound=Union[ArrayLike, torch.Tensor])


class FilterProtocol(Protocol, Generic[T]):

    # discrete_observation_dim: ReadOnlyDescriptor[int]

    def update(self, observation: T) -> None: ...

    def estimate(self) -> T: ...


def compute_loss(
    filter: FilterProtocol,
    observation_trajectory: NDArray,
    discrete_observation_dim: int,
) -> float:
    """
    Compute the empirical optimal loss of a given HMM filter.

    Parameters
    ----------
    filter : HiddenMarkovModelFilter
        The filter to be used for the estimation.
    observation_trajectory : NDArray
        shape = (batch_size, 1, time_horizon)

    Returns
    -------
    loss_mean : float
        The average loss of the optimal estimation.
    """
    logger.info("")
    logger.info("Computing the empirical loss of a given HMM filter...")

    batch_size, _, time_horizon = observation_trajectory.shape
    loss_trajectory = np.empty((batch_size, time_horizon - 1))
    for k, (observation, next_observation_one_hot) in logger.progress_bar(
        enumerate(
            observation_generator(
                observation_trajectory=observation_trajectory,
                discrete_observation_dim=discrete_observation_dim,
            )
        ),
        total=time_horizon - 1,
    ):
        if isinstance(filter, LearningHmmFilter):
            with BaseLearningProcess.inference_mode(filter):
                filter.update(torch.tensor(observation))
                estimation = filter.estimate().numpy()
        else:
            filter.update(observation)
            estimation = filter.estimate()

        loss_trajectory[:, k] = cross_entropy(
            input_probability=estimation,
            target_probability=next_observation_one_hot,
        )
    loss_mean = float(loss_trajectory.mean())
    return loss_mean


# def compute_layer_loss_trajectory(
#     learning_filter: LearningHmmFilter,
#     observation_trajectory: NDArray,
# ) -> NDArray:
#     """
#     Compute the loss of the learning_filter over the observation_trajectory.

#     Parameters
#     ----------
#     learning_filter : LearningHiddenMarkovModelFilter
#         The learning filter to be used for the estimation.
#     observation_trajectory : NDArray
#         shape = (batch_size, 1, time_horizon)

#     Returns
#     -------
#     loss_trajectory : NDArray
#         The loss trajectory of the learning_filter for each layer.
#         shape = (batch_size, layer_dim, time_horizon - 1)
#     loss_mean_over_layers : NDArray
#         The average loss of the learning_filter for each layer.
#         shape = (layer_dim,)
#     """

#     logger.info("")
#     logger.info(
#         "Computing the average loss of the learning_filter over layers"
#     )

#     if observation_trajectory.ndim == 2:
#         observation_trajectory = observation_trajectory[np.newaxis, ...]
#     batch_size, _, time_horizon = observation_trajectory.shape
#     layer_dim = learning_filter.layer_dim
#     loss_trajectory = np.empty(
#         (batch_size, layer_dim, time_horizon - 1)
#     )
#     # learning_filter.estimation_option = (
#     #     EstimationConfig.Option.PREDICTED_NEXT_OBSERVATION_PROBABILITY_OVER_LAYERS
#     # )

#     with BaseLearningProcess.inference_mode(learning_filter):
#         # observation_queue = FixLengthObservationQueue(
#         #     max_length=128,
#         # )
#         learning_filter.reset()
#         for k, (observation, next_observation) in logger.progress_bar(
#             enumerate(
#                 observation_generator(
#                     observation_trajectory=observation_trajectory,
#                     discrete_observation_dim=learning_filter.discrete_observation_dim,
#                 )
#             ),
#             total=time_horizon - 1,
#         ):
#             # learning_filter.reset()
#             # observation_queue.append(observation)
#             # learning_filter.update(torch.tensor(observation_queue.to_numpy()))

#             learning_filter.update(torch.tensor(observation))
#             learning_filter.estimate()
#             layer_output = learning_filter.estimation.numpy()
#             # predicted_next_observation_probability = (
#             #     learning_filter.predicted_next_observation_probability.numpy()
#             # )
#             # for i in range(2048):
#             #     assert np.allclose(
#             #         layer_output[i, 1, :],
#             #         predicted_next_observation_probability[i]
#             #     ), (layer_output[i, 1, :], predicted_next_observation_probability[i])
#             for l in range(layer_dim):
#                 loss_trajectory[:, l, k] = cross_entropy(
#                     input_probability=layer_output[:, l, :],
#                     target_probability=next_observation,
#                 )
#     loss_mean_over_layers: NDArray = np.mean(loss_trajectory, axis=(0, 2))

#     return loss_mean_over_layers


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
    logger.info("")
    logger.info(
        "Computing an example loss trajectory of the filter and learning_filter"
    )
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

            # learning_filter.reset()
            # observation_queue.append(observation[np.newaxis, :])
            # learning_filter.update(torch.tensor(observation_queue.to_numpy()))

            filter.update(observation)
            estimation = filter.estimate()

            filter_estimated_next_observation_probability[:, k] = estimation
            filter_loss_trajectory[k] = cross_entropy(
                input_probability=estimation[np.newaxis, :],
                target_probability=next_observation[np.newaxis, :],
            )[0]

            learning_filter.update(torch.tensor(observation))
            estimation = learning_filter.estimate().numpy()
            learning_filter_estimated_next_observation_probability[:, k] = (
                estimation
            )

            learning_filter_loss_trajectory[k] = cross_entropy(
                input_probability=estimation[np.newaxis, :],
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
