from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from ss.estimation.filtering.hmm import HmmFilter
from ss.estimation.filtering.hmm.learning.module import LearningHmmFilter
from ss.system.markov import HiddenMarkovModel
from ss.utility.data import Data
from ss.utility.device import DeviceManager
from ss.utility.learning.process import BaseLearningProcess
from ss.utility.learning.process.checkpoint import CheckpointInfo
from ss.utility.logging import Logging

from . import figure as Figure
from .utility import (
    compute_layer_loss_trajectory,
    compute_loss_trajectory,
    compute_optimal_loss,
    get_estimation_model,
)

logger = Logging.get_logger(__name__)


def analysis(
    data_filepath: Path,
    model_folderpath: Path,
    model_filename: Path,
    result_directory: Path,
) -> None:
    model_filepath = model_folderpath / model_filename

    DeviceManager()

    # Prepare data
    data = Data.load(data_filepath)
    time_trajectory: NDArray = np.array(data["time"])
    observation_trajectory: NDArray = np.array(
        data["observation"], dtype=np.int64
    )  # (number_of_systems, 1, time_horizon)
    number_of_systems = int(data.meta_info["number_of_systems"])
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

    # Load the module
    module_filename = model_filepath.with_suffix(
        LearningHmmFilter.FILE_EXTENSIONS[0]
    )
    learning_filter, _ = LearningHmmFilter.load(
        module_filename,
        safe_callables={
            torch.nn.functional.cross_entropy,
            torch.optim.AdamW,
            # types of extra arguments
        },
    )

    np.set_printoptions(precision=3, suppress=True)
    with BaseLearningProcess.inference_mode(learning_filter):
        emission_matrix = learning_filter.emission_matrix.numpy()
        logger.info("(layer 0) learned emission matrix = ")
        for k in range(emission_matrix.shape[0]):
            logger.info(f"    {emission_matrix[k]}")
        transition_matrix = learning_filter.transition_matrix[0][0].numpy()

    Figure.StochasticMatrixFigure(
        stochastic_matrix=transition_matrix,
        fig_title="Transition Matrix",
    ).plot()

    Figure.StochasticMatrixFigure(
        stochastic_matrix=emission_matrix,
        fig_title="Emission Matrix",
    ).plot()

    # Convert the natural logarithm to the log_base logarithm
    log_base = np.e
    scaling = 1.0 / np.log(log_base)

    # Compute the loss trajectory of the filter and learning_filter
    logger.info("")
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

    logger.info("")
    logger.info(
        "Computing the average loss of the learning_filter over layers"
    )

    _, loss_mean_over_layer = compute_layer_loss_trajectory(
        learning_filter=learning_filter,
        observation_trajectory=observation_trajectory,
    )
    loss_mean_over_layer = loss_mean_over_layer * scaling
    logger.info(f"empirical average loss (over layers):")
    for l, loss in enumerate(loss_mean_over_layer):
        logger.info(f"    layer {l}: {float(loss)}")

    # Compute the random guess loss
    random_guess_loss = -np.log(1 / discrete_observation_dim) * scaling

    # Compute the empirical optimal loss
    logger.info("")
    logger.info(
        "Computing the empirical optimal loss (the cross-entropy loss of the hmm-filter)"
    )
    number_of_systems = int(data.meta_info["number_of_systems"])
    empirical_optimal_loss = float(
        compute_optimal_loss(
            filter.duplicate(number_of_systems),
            observation_trajectory,
        )
        * scaling
    )
    logger.info(f"{empirical_optimal_loss = }")

    # Plot the training and validation loss together with the optimal loss
    checkpoint_info = CheckpointInfo.load(model_filepath.with_suffix(".hdf5"))
    loss_figure = Figure.IterationFigure(
        training_loss_history=checkpoint_info["__training_loss_history__"],
        validation_loss_history=checkpoint_info["__validation_loss_history__"],
        scaling=scaling,
    ).plot()
    Figure.add_loss_line(
        loss_figure.loss_plot_ax,
        random_guess_loss,
        "random guess loss: {:.3f}",
        log_base=log_base,
    )
    Figure.add_loss_line(
        loss_figure.loss_plot_ax,
        empirical_optimal_loss,
        "optimal loss: {:.3f}\n(based on HMM-filter)",
        log_base=log_base,
        text_offset=(64, -48),
    )
    for l, loss in enumerate(loss_mean_over_layer):
        Figure.add_loss_line(
            loss_figure.loss_plot_ax,
            loss,
            f"loss on layer {l}" + ": {:.3f}",
            log_base=log_base,
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
        loss_scaling=scaling,
    ).plot()

    Figure.show()
