from pathlib import Path

import numpy as np
import torch

from ss.estimation.filtering.hmm.learning import config as Config
from ss.estimation.filtering.hmm.learning import dataset as Dataset
from ss.estimation.filtering.hmm.learning import module as Module
from ss.estimation.filtering.hmm.learning import process as Process
from ss.utility.data import Data
from ss.utility.device import DeviceManager
from ss.utility.learning.process.config import TrainingConfig
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


def train(
    data_filepath: Path,
    model_filepath: Path,
    result_directory: Path,
) -> None:
    device_manager = DeviceManager()

    # Prepare data
    data = Data.load(data_filepath)
    observation = data["observation"]
    number_of_systems = int(data.meta_info["number_of_systems"])

    (
        training_loader,
        evaluation_loader,
        testing_loader,
    ) = (
        Dataset.HmmObservationDataset(
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

    # Prepare model
    discrete_observation_dim = int(data.meta_info["discrete_observation_dim"])
    discrete_state_dim = int(np.sqrt(data.meta_info["discrete_state_dim"]))
    config = Config.LearningHmmFilterConfig.basic_config(
        state_dim=discrete_state_dim,
        discrete_observation_dim=discrete_observation_dim,
        feature_dim_over_layers=(1,),
    )
    config.dropout.rate = 0.05
    config.dropout.log_zero_scale = -10.0
    config.transition.matrix.option = (
        config.transition.matrix.Option.FULL_MATRIX
    )
    config.transition.matrix.initializer = (
        config.transition.matrix.Initializer.NORMAL_DISTRIBUTION
    )
    # config.transition.matrix.initializer.variance = 0.0
    config.transition.initial_state.initializer = (
        config.transition.initial_state.Initializer.NORMAL_DISTRIBUTION
    )
    # config.transition.initial_state.initializer.variance = 0.0
    config.emission.matrix.initializer = (
        config.emission.matrix.Initializer.NORMAL_DISTRIBUTION
    )
    # config.emission.matrix.initializer.variance = 0.0
    config.estimation.option = (
        config.estimation.Option.PREDICTED_NEXT_OBSERVATION_PROBABILITY
    )
    # config.transition.skip_first_transition = True

    learning_filter = Module.LearningHmmFilter(config)

    # Prepare loss function
    loss_function = torch.nn.functional.cross_entropy

    # Prepare optimizer
    optimizer = torch.optim.AdamW(
        learning_filter.parameters(), lr=0.0005, weight_decay=0.01
    )

    # Train model
    learning_process = Process.LearningHmmFilterProcess(
        model=learning_filter,
        loss_function=loss_function,
        optimizer=optimizer,
    )
    training_config = TrainingConfig()
    training_config.evaluation.per_iteration_period = 100
    training_config.termination.max_epoch = 5
    training_config.checkpoint.filepath = model_filepath.with_suffix("")
    training_config.checkpoint.per_epoch_period = 1
    training_config.checkpoint.appendix.option = (
        training_config.checkpoint.appendix.Option.COUNTER
    )
    with device_manager.monitor_performance(
        sampling_rate=10.0,
        result_directory=result_directory,
    ):
        learning_process.train(
            training_loader, evaluation_loader, training_config
        )

    # Test model
    learning_process.test_model(testing_loader)
