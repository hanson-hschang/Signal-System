from typing import Any

from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray
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

from ._learning_hmm_filtering_figure import add_optimal_loss_line
from ._learning_hmm_filtering_utility import (
    ObservationDataset,
    cross_entropy,
    data_split,
    get_observation_model,
    observation_generator,
)

logger = Logging.get_logger(__name__)


class LearningHMMFilterProcess(BaseLearningProcess):
    def _evaluate_one_batch(self, data_batch: Any) -> torch.Tensor:
        (
            observation_trajectory,
            next_observation_trajectory,
        ) = ObservationDataset.from_batch(
            data_batch
        )  # (batch_size, max_length), (batch_size, max_length)
        estimated_next_observation_probability_trajectory = self._model(
            observation_trajectory=observation_trajectory
        )  # (batch_size, max_length, observation_dim)
        _loss: torch.Tensor = self._loss_function(
            torch.moveaxis(
                estimated_next_observation_probability_trajectory, 1, 2
            ),  # (batch_size, observation_dim, max_length)
            next_observation_trajectory,  # (batch_size, max_length)
        )
        return _loss


def train(
    data_filepath: Path,
    model_filepath: Path,
) -> None:
    # Prepare data
    data = Data.load(data_filepath)
    emission_probability_matrix: NDArray = np.array(
        data.meta_data["emission_probability_matrix"]
    )
    state_dim = emission_probability_matrix.shape[0]
    observation = data["observation"]
    number_of_systems = int(data.meta_info["number_of_systems"])
    discrete_observation_dim = int(data.meta_info["discrete_observation_dim"])
    training_loader, evaluation_loader, testing_loader = data_split(
        observation=observation,
        split_ratio=[0.7, 0.2, 0.1],
        number_of_systems=number_of_systems,
    )

    # Prepare model
    params = LearningHiddenMarkovModelFilterParameters(
        state_dim=state_dim,  # similar to embedding dimension in the transformer
        observation_dim=discrete_observation_dim,  # similar to number of tokens in the transformer
        feature_dim=1,  # similar to number of heads in the transformer
        layer_dim=1,  # similar to number of layers in the transformer
        dropout_rate=0.2,  # similar to dropout rate in the transformer
    )
    filter = LearningHiddenMarkovModelFilter(params)
    filter.set_emission_matrix(emission_probability_matrix, trainable=False)

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
        model_filename=model_filepath,
        save_model_epoch_skip=1,
    )
    learning_process.train(training_loader, evaluation_loader)

    # Test model
    learning_process.test_model(testing_loader)


def visualization(
    data_filepath: Path,
    model_filepath: Path,
) -> None:
    # Prepare data
    data = Data.load(data_filepath)
    time_horizon = data["time"].shape[-1]
    observation_trajectory: NDArray = np.array(
        data["observation"], dtype=np.int64
    )  # (number_of_systems, 1, time_horizon)
    discrete_observation_dim = int(data.meta_info["discrete_observation_dim"])

    # Prepare filter
    number_of_systems = int(data.meta_info["number_of_systems"])
    transition_probability_matrix: NDArray = np.array(
        data.meta_data["transition_probability_matrix"]
    )
    emission_probability_matrix: NDArray = np.array(
        data.meta_data["emission_probability_matrix"]
    )
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

    # Load the model
    filter = LearningHiddenMarkovModelFilter.load(model_filepath)

    # Estimate the optimal loss
    logger.info(
        "Start estimating the optimal loss (the cross-entropy loss of the hmm-filter)"
    )
    loss_trajectory = []
    with logging_redirect_tqdm(loggers=[logger]):
        for observation, next_observation in tqdm(
            observation_generator(
                observation_trajectory,
                discrete_observation_dim=discrete_observation_dim,
            ),
            total=time_horizon - 1,
        ):
            estimator.update(observation=observation)
            estimator.estimate()
            loss_trajectory.append(
                cross_entropy(
                    input_probability=estimator.estimation,
                    target_probability=next_observation,
                )
            )
    average_loss = float(np.mean(loss_trajectory))
    logger.info(f"{average_loss=}")

    with torch.no_grad():
        logger.info(f"{emission_probability_matrix=}")
        logger.info(f"\n{filter.emission_matrix=}")

    # Plot the training and validation loss together with the optimal loss
    checkpoint_info = CheckpointInfo.load(model_filepath.with_suffix(".hdf5"))
    fig = IterationFigure(
        training_loss_trajectory=checkpoint_info["training_loss_history"],
        validation_loss_trajectory=checkpoint_info["evaluation_loss_history"],
    ).plot()
    add_optimal_loss_line(fig.loss_plot_ax, average_loss)
    plt.show()


def inference(
    data_filepath: Path,
    model_filepath: Path,
) -> None:
    # Prepare data
    data = Data.load(data_filepath)
    number_of_systems = int(data.meta_info["number_of_systems"])
    observation_trajectory: NDArray = np.array(
        (
            data["observation"]
            if number_of_systems == 1
            else data["observation"][0]
        ),
        dtype=np.int64,
    )[
        0
    ]  # (time_horizon,)

    # Load the model
    filter = LearningHiddenMarkovModelFilter.load(model_filepath)
    with torch.no_grad():
        logger.info(f"\n{filter.emission_matrix=}")

    # Inference
    given_time_horizon = 20  # This is like prompt length
    future_time_steps = 5  # This is like how many next token to predict
    number_of_samples = 5  # This is like number of answers to generate based on the same prompt (test-time compute)
    with Mode.inference(filter):
        filter.update(
            torch.tensor(
                observation_trajectory[given_time_horizon], dtype=torch.int64
            )
        )
        logger.info(
            f"The sequence of the next {future_time_steps} observations from the data is:\n"
            f"{observation_trajectory[given_time_horizon + 1: given_time_horizon + 1 + future_time_steps]}"
        )
        for k in range(future_time_steps):
            next_observation = torch.tensor(
                observation_trajectory[given_time_horizon + k + 1],
                dtype=torch.int64,
            )
            logger.info(
                "          next observation = "
                f"{next_observation.detach().numpy()}"
            )
            filter.estimate()
            estimated_next_observation_probability = (
                filter.estimated_next_observation_probability
            )
            predicted_next_observation = torch.multinomial(
                estimated_next_observation_probability, 1
            )
            logger.info(
                "predicted next observation = "
                f"{predicted_next_observation[0].detach().numpy()}"
            )
            logger.info(
                "with probability distribution = \n"
                f"{estimated_next_observation_probability.detach().numpy()}"
            )
            filter.update(next_observation)
