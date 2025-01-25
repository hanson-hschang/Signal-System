from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from ss.estimation.filtering.hmm_filtering import (
    HiddenMarkovModelFilter,
    HiddenMarkovModelObservationDataset,
    LearningHiddenMarkovModelFilter,
    LearningHiddenMarkovModelFilterBlockOption,
    LearningHiddenMarkovModelFilterConfig,
    LearningHiddenMarkovModelFilterProcess,
    hmm_observation_data_split_to_loaders,
)
from ss.learning import CheckpointInfo, IterationFigure, Mode
from ss.system.markov import HiddenMarkovModel
from ss.utility.data import Data
from ss.utility.logging import Logging

from ._learning_hmm_filtering_figure import (
    FilterResultFigure,
    add_loss_line,
    update_loss_ylim,
)
from ._learning_hmm_filtering_utility import (
    compute_loss_trajectory,
    compute_optimal_loss,
    get_estimation_model,
)

logger = Logging.get_logger(__name__)


def train(
    data_filepath: Path,
    model_filepath: Path,
) -> None:
    # Prepare data
    data = Data.load(data_filepath)
    observation = data["observation"]
    number_of_systems = int(data.meta_info["number_of_systems"])

    (
        training_loader,
        evaluation_loader,
        testing_loader,
    ) = (
        HiddenMarkovModelObservationDataset(
            observation=observation,
            number_of_systems=number_of_systems,
            max_length=256,
            stride=64,
        )
        .split(
            split_ratio=[0.7, 0.2, 0.1],
            random_seed=2025,
        )
        .to_loaders(
            batch_size=128,
            shuffle=True,
        )
    )

    # (
    #     training_loader,
    #     evaluation_loader,
    #     testing_loader,
    # ) = hmm_observation_data_split_to_loaders(
    #     observation=observation,
    #     number_of_systems=number_of_systems,
    #     max_length=256,
    #     stride=64,
    #     split_ratio=[0.7, 0.2, 0.1],
    #     random_seed=2025,
    # )

    # Prepare model
    discrete_observation_dim = int(data.meta_info["discrete_observation_dim"])
    discrete_state_dim = int(np.sqrt(data.meta_info["discrete_state_dim"]))
    config = LearningHiddenMarkovModelFilterConfig(
        state_dim=discrete_state_dim,  # similar to embedding dimension in the transformer
        discrete_observation_dim=discrete_observation_dim,  # similar to number of tokens in the transformer
        feature_dim=1,  # similar to number of heads in the transformer
        layer_dim=4,  # similar to number of layers in the transformer
        dropout_rate=0.2,  # similar to dropout rate in the transformer
        block_option=LearningHiddenMarkovModelFilterBlockOption.SPATIAL_INVARIANT,
    )
    learning_filter = LearningHiddenMarkovModelFilter(config)

    # Prepare loss function
    loss_function = torch.nn.functional.cross_entropy

    # Prepare optimizer
    optimizer = torch.optim.AdamW(
        learning_filter.parameters(), lr=0.0005, weight_decay=0.01
    )

    # Train model
    learning_process = LearningHiddenMarkovModelFilterProcess(
        model=learning_filter,
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
    time_trajectory: NDArray = np.array(data["time"])
    time_horizon = time_trajectory.shape[-1]
    observation_trajectory: NDArray = np.array(
        data["observation"], dtype=np.int64
    )  # (number_of_systems, 1, time_horizon)
    discrete_observation_dim = int(data.meta_info["discrete_observation_dim"])
    transition_matrix: NDArray = np.array(data.meta_data["transition_matrix"])
    emission_matrix: NDArray = np.array(data.meta_data["emission_matrix"])

    # Prepare HMM filter
    filter = HiddenMarkovModelFilter(
        system=HiddenMarkovModel(
            transition_matrix=transition_matrix,
            emission_matrix=emission_matrix,
        ),
        estimation_model=get_estimation_model(
            transition_matrix=transition_matrix,
            emission_matrix=emission_matrix,
            future_time_steps=1,
        ),
    )

    # Load the model
    learning_filter = LearningHiddenMarkovModelFilter.load(model_filepath)

    # Compute the loss trajectory of the filter and learning_filter
    logger.info(
        "Computing the loss trajectory of the filter and learning_filter"
    )
    max_time_horizon = 200
    time_slice = slice(min(time_horizon, max_time_horizon))
    time_trajectory_test = time_trajectory[time_slice]
    observation_trajectory_test = observation_trajectory[0, :, time_slice]
    filter_result_trajectory, learning_filter_result_trajectory = (
        compute_loss_trajectory(
            filter=filter,
            learning_filter=learning_filter,
            observation_trajectory=observation_trajectory_test,
        )
    )

    # Compute the random guess loss
    random_guess_loss = -np.log(1 / discrete_observation_dim)

    # Compute the empirical optimal loss
    logger.info(
        "Computing the empirical optimal loss (the cross-entropy loss of the hmm-filter)"
    )
    number_of_systems = int(data.meta_info["number_of_systems"])
    empirical_optimal_loss = compute_optimal_loss(
        filter.duplicate(number_of_systems),
        observation_trajectory,
    )
    logger.info(f"{empirical_optimal_loss = }")

    # Plot the training and validation loss together with the optimal loss
    checkpoint_info = CheckpointInfo.load(model_filepath.with_suffix(".hdf5"))
    loss_figure = IterationFigure(
        training_loss_trajectory=checkpoint_info["training_loss_history"],
        validation_loss_trajectory=checkpoint_info["evaluation_loss_history"],
    ).plot()
    add_loss_line(
        loss_figure.loss_plot_ax,
        random_guess_loss,
        "random guess loss: {:.2f}",
    )
    add_loss_line(
        loss_figure.loss_plot_ax,
        empirical_optimal_loss,
        "optimal loss: {:.2f}\n(based on HMM-filter)",
    )
    update_loss_ylim(
        loss_figure.loss_plot_ax, (empirical_optimal_loss, random_guess_loss)
    )

    # Plot the filter result comparison
    result_figure = FilterResultFigure(
        time_trajectory=time_trajectory_test,
        observation_trajectory=observation_trajectory_test,
        filter_result_trajectory_dict=dict(
            filter=filter_result_trajectory,
            learning_filter=learning_filter_result_trajectory,
        ),
    ).plot()
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
            data["observation"][0]
            if number_of_systems == 1
            else data["observation"][0, 0]
        ),
        dtype=np.int64,
    )  # (time_horizon,)

    np.set_printoptions(precision=3)

    # Load the model
    learning_filter = LearningHiddenMarkovModelFilter.load(model_filepath)
    with torch.no_grad():
        emission_matrix = learning_filter.emission_matrix.numpy()
        logger.info("emission_matrix = ")
        for k in range(emission_matrix.shape[0]):
            logger.info(f"    {emission_matrix[k]}")

    # Inference
    given_time_horizon = 20  # This is like prompt length
    future_time_steps = 5  # This is like how many next token to predict
    number_of_samples = 5  # This is like number of answers to generate based on the same prompt (test-time compute)

    with Mode.inference(learning_filter):
        _observation_trajectory = torch.tensor(
            observation_trajectory[:given_time_horizon], dtype=torch.int64
        )
        logger.info(
            f"The sequence of the first {given_time_horizon} observations from the data is: "
            f"{observation_trajectory[:given_time_horizon]}"
        )
        learning_filter.update(_observation_trajectory)
        logger.info(
            f"The sequence of the next {future_time_steps} observations from the data is: "
            f"{observation_trajectory[given_time_horizon + 1: given_time_horizon + 1 + future_time_steps]}"
        )
        for k in range(future_time_steps):
            next_observation = torch.tensor(
                observation_trajectory[given_time_horizon + k + 1],
                dtype=torch.int64,
            )
            logger.info("")
            logger.info(
                "          next observation = " f"{next_observation.detach()}"
            )
            predicted_next_observation_probability = (
                learning_filter.predicted_next_observation_probability.numpy()
            )
            predicted_next_observation = learning_filter.predict()
            logger.info(
                "predicted next observation = "
                f"{predicted_next_observation[0]}"
            )
            logger.info(
                "with probability distribution(%) = "
                f"{predicted_next_observation_probability*100}"
            )
            learning_filter.update(next_observation)
