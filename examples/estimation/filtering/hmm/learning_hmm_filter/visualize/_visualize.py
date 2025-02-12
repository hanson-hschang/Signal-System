from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from ss.estimation.filtering.hmm import HmmFilter
from ss.estimation.filtering.hmm.learning import config as Config
from ss.estimation.filtering.hmm.learning import module as Module
from ss.system.markov import HiddenMarkovModel
from ss.utility.data import Data
from ss.utility.learning.process import CheckpointInfo
from ss.utility.logging import Logging

from . import figure as Figure
from .utility import (
    compute_layer_loss_trajectory,
    compute_loss_trajectory,
    compute_optimal_loss,
    get_estimation_model,
)

logger = Logging.get_logger(__name__)


def visualize(
    data_filepath: Path,
    model_filepath: Path,
) -> None:
    # Prepare data
    data = Data.load(data_filepath)
    time_trajectory: NDArray = np.array(data["time"])
    observation_trajectory: NDArray = np.array(
        data["observation"], dtype=np.int64
    )  # (number_of_systems, 1, time_horizon)
    discrete_observation_dim = int(data.meta_info["discrete_observation_dim"])
    transition_matrix: NDArray = np.array(data.meta_data["transition_matrix"])
    emission_matrix: NDArray = np.array(data.meta_data["emission_matrix"])

    # Prepare HMM filter
    filter = HmmFilter(
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
    learning_filter, _ = Module.LearningHmmFilter.load(model_filepath)
    np.set_printoptions(precision=3)
    with torch.no_grad():
        emission_matrix = learning_filter.emission_matrix.numpy()
        logger.info("learned emission_matrix = ")
        for k in range(emission_matrix.shape[0]):
            logger.info(f"    {emission_matrix[k]}")
    learning_filter.set_estimation_option(
        Config.EstimationConfig.Option.PREDICTED_NEXT_OBSERVATION_PROBABILITY_OVER_LAYERS
    )

    # Compute the loss trajectory of the filter and learning_filter
    logger.info(
        "Computing an example loss trajectory of the filter and learning_filter"
    )
    example_time_trajectory = time_trajectory
    example_observation_trajectory = observation_trajectory[0]
    filter_result_trajectory, learning_filter_result_trajectory = (
        compute_loss_trajectory(
            filter=filter,
            learning_filter=learning_filter,
            observation_trajectory=example_observation_trajectory,
        )
    )
    logger.info(
        "Computing the average loss of the learning_filter over layers"
    )
    learning_filter.reset()
    _, average_loss_over_layer = compute_layer_loss_trajectory(
        learning_filter=learning_filter,
        observation_trajectory=observation_trajectory,
    )
    logger.info(f"{average_loss_over_layer = }")

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
    loss_figure = Figure.IterationFigure(
        training_loss_trajectory=checkpoint_info["training_loss_history"],
        validation_loss_trajectory=checkpoint_info["evaluation_loss_history"],
    ).plot()
    Figure.add_loss_line(
        loss_figure.loss_plot_ax,
        random_guess_loss,
        "random guess loss: {:.3f}",
    )
    Figure.add_loss_line(
        loss_figure.loss_plot_ax,
        empirical_optimal_loss,
        "optimal loss: {:.3f}\n(based on HMM-filter)",
    )
    for l, loss in enumerate(average_loss_over_layer):
        Figure.add_loss_line(
            loss_figure.loss_plot_ax,
            loss,
            f"loss on layer {l}" + ": {:.3f}",
        )
    Figure.update_loss_ylim(
        loss_figure.loss_plot_ax, (empirical_optimal_loss, random_guess_loss)
    )

    # Plot the filter result comparison
    Figure.FilterResultFigure(
        time_trajectory=example_time_trajectory,
        observation_trajectory=example_observation_trajectory,
        filter_result_trajectory_dict=dict(
            filter=filter_result_trajectory,
            learning_filter=learning_filter_result_trajectory,
        ),
    ).plot()
    Figure.show()
