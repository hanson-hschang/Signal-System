from typing import Any

from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ss.estimation.filtering.hmm_filtering import (
    HiddenMarkovModelFilter,
    LearningHiddenMarkovModelFilter,
    LearningHiddenMarkovModelFilterParameters,
)
from ss.learning import (
    BaseLearningProcess,
    CheckpointInfo,
    IterationFigure,
    Mode,
)
from ss.system.markov import HiddenMarkovModel
from ss.utility.data import Data
from ss.utility.logging import Logging

from ._learning_hmm_filtering_utility import (
    ObservationDataset,
    add_optimal_loss,
    cross_entropy,
    data_split,
    get_observation_model,
    observation_generator,
)

logger = Logging.get_logger(__name__)


class LearningHMMFilterProcess(BaseLearningProcess):

    def _evaluate_one_batch(self, data_batch: Any) -> torch.Tensor:
        observation_trajectory, next_observation_trajectory = (
            ObservationDataset.from_batch(data_batch)
        )  # (batch_size, max_length), (batch_size, max_length)
        estimated_next_observation_probability_trajectory = self._model(
            observation_trajectory=observation_trajectory
        )  # (batch_size, max_length, observation_dim)
        _loss = self._loss_function(
            torch.moveaxis(
                estimated_next_observation_probability_trajectory, 1, 2
            ),  # (batch_size, observation_dim, max_length)
            next_observation_trajectory,  # (batch_size, max_length)
        )
        return _loss


def train(
    data_filename: Path,
    result_directory: Path,
    model_filename: Path,
) -> None:
    # Prepare data
    data = Data.load(data_filename)
    observation = data["observation_value"]
    number_of_systems = int(data.meta_info["number_of_systems"])
    observation_dim = int(data.meta_info["observation_dim"])
    training_loader, evaluation_loader, testing_loader = data_split(
        observation=observation,
        split_ratio=[0.7, 0.2, 0.1],
        number_of_systems=number_of_systems,
    )

    # Prepare model
    params = LearningHiddenMarkovModelFilterParameters(
        state_dim=3,  # similar to embedding dimension in the transformer
        observation_dim=observation_dim,  # similar to number of tokens in the transformer
        feature_dim=1,  # similar to number of heads in the transformer
        layer_dim=1,  # similar to number of layers in the transformer
        dropout_rate=0.2,  # similar to dropout rate in the transformer
    )
    filter = LearningHiddenMarkovModelFilter(params)

    # Prepare loss function
    loss_function = torch.nn.functional.cross_entropy

    # Prepare optimizer
    optimizer = torch.optim.AdamW(
        filter.parameters(), lr=0.0005, weight_decay=0.01
    )

    # Train model
    learning_process = LearningHMMFilterProcess(
        model=filter,
        loss_function=loss_function,
        optimizer=optimizer,
        number_of_epochs=5,
        model_filename=result_directory / model_filename,
        save_model_epoch_skip=1,
    )
    learning_process.train(training_loader, evaluation_loader)

    # Test model
    learning_process.test_model(testing_loader)


def visualization(
    data_filename: Path,
    result_directory: Path,
    model_filename: Path,
) -> None:

    # Prepare data
    data = Data.load(data_filename)
    time_horizon = data["time"].shape[-1]
    observation_trajectory = data[
        "observation"
    ]  # (number_of_systems, observation_dim, time_horizon)

    # Prepare filter
    number_of_systems = int(data.meta_info["number_of_systems"])
    transition_probability_matrix = data.meta_data[
        "transition_probability_matrix"
    ]
    emission_probability_matrix = data.meta_data["emission_probability_matrix"]
    estimator = HiddenMarkovModelFilter(
        system=HiddenMarkovModel(
            transition_probability_matrix=transition_probability_matrix,
            emission_probability_matrix=emission_probability_matrix,
            number_of_systems=number_of_systems,
        ),
        estimation_model=get_observation_model(
            transition_probability_matrix=transition_probability_matrix,
            emission_probability_matrix=emission_probability_matrix,
            future_time_steps=1,
        ),
    )

    # Estimate the optimal loss
    logger.info(
        "Start estimating the optimal loss (the cross-entropy loss of the hmm-filter)"
    )
    loss_trajectory = []
    with logging_redirect_tqdm(loggers=[logger]):
        for observation, next_observation in tqdm(
            observation_generator(observation_trajectory),
            total=time_horizon - 1,
        ):
            estimator.update(observation=observation)
            estimator.estimate()
            loss_trajectory.append(
                cross_entropy(
                    input_probability=estimator.estimated_function_value,
                    target_probability=next_observation,
                )
            )
    average_loss = float(np.mean(loss_trajectory))
    logger.info(f"{average_loss=}")

    # Load the model
    model_filename = result_directory / model_filename
    filter = LearningHiddenMarkovModelFilter.load(model_filename)
    with torch.no_grad():
        logger.info(f"\n{filter.emission_matrix=}")

    # Plot the training and validation loss together with the optimal loss
    checkpoint_info = CheckpointInfo.load(model_filename.with_suffix(".hdf5"))
    fig = IterationFigure(
        training_loss_trajectory=checkpoint_info["training_loss_history"],
        validation_loss_trajectory=checkpoint_info["evaluation_loss_history"],
    ).plot()
    add_optimal_loss(fig.loss_plot_ax, average_loss)
    plt.show()


def inference(
    result_directory: Path,
    model_filename: Path,
) -> None:
    # Load the model
    model_filename = result_directory / model_filename
    filter = LearningHiddenMarkovModelFilter.load(model_filename)
    with torch.no_grad():
        logger.info(f"\n{filter.emission_matrix=}")

    # Inference
    with Mode.inference(filter):
        observation_trajectory = torch.tensor(
            [0, 1, 2, 4, 1, 5, 1, 1, 1, 0, 2, 0, 3, 2, 3, 1, 6, 1],
        )
        filter.update(observation_trajectory)
        for _ in range(5):
            filter.estimate()
            estimated_next_observation_probability = (
                filter.estimated_next_observation_probability
            )
            logger.info(estimated_next_observation_probability)
            predicted_next_observation = torch.multinomial(
                estimated_next_observation_probability, 1
            )
            logger.info(predicted_next_observation)
            filter.update(predicted_next_observation)
