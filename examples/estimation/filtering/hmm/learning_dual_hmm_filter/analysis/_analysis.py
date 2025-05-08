from typing import cast

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from ss.estimation.dual_filtering.hmm.learning.module import (
    LearningDualHmmFilter,
)
from ss.estimation.filtering.hmm import HmmFilter
from ss.system.markov import HiddenMarkovModel
from ss.utility.data import Data
from ss.utility.device.manager import DeviceManager
from ss.utility.learning.parameter.transformer.softmax import (
    SoftmaxTransformer,
)
from ss.utility.learning.parameter.transformer.softmax.config import (
    SoftmaxTransformerConfig,
)
from ss.utility.learning.process.checkpoint import CheckpointInfo
from ss.utility.logging import Logging

from . import figure as Figure
from . import utility as Utility

logger = Logging.get_logger(__name__)


def analysis(
    data_filepath: Path,
    model_folderpath: Path,
    model_filename: str,
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
        estimation_model=Utility.get_estimation_model(
            transition_matrix=transition_matrix,
            emission_matrix=emission_matrix,
            future_time_steps=0,
        ),
    )

    # Load module
    module_filename = model_filepath.with_suffix(
        LearningDualHmmFilter.FILE_EXTENSIONS[0]
    )
    learning_filter, _ = LearningDualHmmFilter[
        SoftmaxTransformer, SoftmaxTransformerConfig
    ].load(
        module_filename,
        safe_callables={
            torch.nn.functional.cross_entropy,
            torch.optim.AdamW,
            # types of extra arguments
        },
    )

    # Display module information
    Utility.module_info(learning_filter)

    # learned_transition_matrix = (
    #     learning_filter.transition_matrix[0].detach().numpy()
    # )
    # learned_emission_matrix = learning_filter.emission_matrix.detach().numpy()

    # Figure.StochasticMatrixFigure(
    #     stochastic_matrix=transition_matrix,
    #     fig_title="Transition Matrix",
    # ).plot()

    # Figure.StochasticMatrixFigure(
    #     stochastic_matrix=emission_matrix,
    #     fig_title="Emission Matrix",
    # ).plot()

    # Figure.StochasticMatrixFigure(
    #     stochastic_matrix=learned_transition_matrix,
    #     fig_title="Learned Transition Matrix",
    # ).plot()

    # Figure.StochasticMatrixFigure(
    #     stochastic_matrix=learned_emission_matrix,
    #     fig_title="Learned Emission Matrix",
    # ).plot()

    # Loss analysis

    ## Convert the natural logarithm to the log_base logarithm
    loss_conversion = Utility.LossConversion()

    ## Compute the random guess loss
    random_guess_loss = loss_conversion(-np.log(1 / discrete_observation_dim))

    ## Compute the empirical optimal loss
    empirical_optimal_loss = loss_conversion(
        Utility.compute_loss(
            filter.duplicate(number_of_systems),
            observation_trajectory,
            filter.discrete_observation_dim,
        )
    )
    logger.info(f"empirical optimal loss = {float(empirical_optimal_loss)}")

    # Compute the empirical loss of the learning_filter
    learning_filter.number_of_systems = number_of_systems
    empirical_learning_filter_loss = loss_conversion(
        Utility.compute_loss(
            learning_filter,
            observation_trajectory[..., :50],
            learning_filter.discrete_observation_dim,
        )
    )
    logger.info(
        f"empirical loss of learned HMM filter = {float(empirical_learning_filter_loss)}"
    )
    # loss_mean_over_layer = loss_conversion(
    #     Utility.compute_layer_loss_trajectory(
    #         learning_filter=learning_filter,
    #         observation_trajectory=observation_trajectory,
    #     )
    # )
    # logger.info(f"empirical average loss (over layers):")
    # for l, loss in enumerate(loss_mean_over_layer):
    #     logger.info(f"    layer {l}: {float(loss)}")

    ## Compute an example loss trajectory of the filter and learning_filter
    learning_filter.number_of_systems = 1
    filter_result_trajectory, learning_filter_result_trajectory = (
        Utility.compute_loss_trajectory(
            filter=filter,
            learning_filter=learning_filter,
            observation_trajectory=(
                example_observation_trajectory := observation_trajectory[0]
            ),
        )
    )

    # Analysis visualization

    ## Plot the training and validation loss together with the optimal loss
    checkpoint_info = CheckpointInfo.load(model_filepath.with_suffix(".hdf5"))
    loss_figure = Figure.IterationFigure(
        training_loss_history=checkpoint_info["__training_loss_history__"],
        validation_loss_history=checkpoint_info["__validation_loss_history__"],
        scaling=loss_conversion.scaling,
        iteration_log_scale=False,
    ).plot()
    Figure.add_loss_line(
        loss_figure.loss_plot_ax,
        random_guess_loss,
        "random guess loss: {:.3f}",
        log_base=loss_conversion.log_base,
    )
    Figure.add_loss_line(
        loss_figure.loss_plot_ax,
        empirical_optimal_loss,
        "optimal loss: {:.3f}\n(based on HMM-filter)",
        log_base=loss_conversion.log_base,
        text_offset=(64, -48),
    )
    Figure.add_loss_line(
        loss_figure.loss_plot_ax,
        empirical_learning_filter_loss,
        "learning filter loss: {:.3f}\n",
        log_base=loss_conversion.log_base,
    )
    # for l, loss in enumerate(loss_mean_over_layer):
    #     Figure.add_loss_line(
    #         loss_figure.loss_plot_ax,
    #         loss,
    #         f"loss on layer {l}" + ": {:.3f}",
    #         log_base=loss_conversion.log_base,
    #     )
    Figure.update_loss_ylim(
        loss_figure.loss_plot_ax, (empirical_optimal_loss, random_guess_loss)
    )

    ## Plot the filter result comparison
    Figure.FilterResultFigure(
        time_trajectory=time_trajectory,
        observation_trajectory=example_observation_trajectory,
        filter_result_trajectory_dict=dict(
            filter=filter_result_trajectory,
            learning_filter=learning_filter_result_trajectory,
        ),
        loss_scaling=loss_conversion.scaling,
    ).plot()

    Figure.show()
