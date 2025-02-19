from typing import Any, Callable, Dict, Union

from pathlib import Path

import torch

from ss.estimation.filtering.hmm.learning import dataset as Dataset
from ss.estimation.filtering.hmm.learning.module import LearningHmmFilter
from ss.estimation.filtering.hmm.learning.module import config as Config
from ss.estimation.filtering.hmm.learning.process import (
    LearningHmmFilterProcess,
)
from ss.utility.data import Data
from ss.utility.device import DeviceManager
from ss.utility.learning.process.checkpoint import CheckpointInfo
from ss.utility.learning.process.config import TrainingConfig
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


def training(
    data_filepath: Path,
    model_folderpath: Path,
    model_filename: Path,
    result_directory: Path,
    new_training: bool = True,
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

    # Prepare module configuration
    config = Config.LearningHmmFilterConfig.basic_config(
        state_dim=int(data.meta_info["discrete_state_dim"]),
        discrete_observation_dim=int(
            data.meta_info["discrete_observation_dim"]
        ),
        feature_dim_over_layers=(1,),
    )
    config.dropout.rate = 0.0
    config.transition.matrix.stochasticizer.temperature.initial_value = 2.0
    config.transition.matrix.stochasticizer.temperature.require_training = True

    # Prepare module
    learning_filter = LearningHmmFilter(config)

    # Define learning process
    class LearningProcess(LearningHmmFilterProcess):
        def __init__(
            self,
            module: LearningHmmFilter,
            loss_function: Callable[..., torch.Tensor],
            optimizer: torch.optim.Optimizer,
            # extra arguments
        ) -> None:
            super().__init__(module, loss_function, optimizer)

        def _save_model_info(self) -> Dict[str, Any]:
            custom_model_info: Dict[str, Any] = dict(
                # save extra arguments
            )
            return custom_model_info

        def _save_checkpoint_info(self) -> CheckpointInfo:
            custom_checkpoint_info = CheckpointInfo(
                # save extra information
            )
            return custom_checkpoint_info

        def _load_checkpoint_info(
            self, checkpoint_info: CheckpointInfo
        ) -> None:
            # load extra information
            pass

    # Prepare learning process
    if new_training:

        # Prepare loss function
        loss_function = torch.nn.functional.cross_entropy

        # Prepare optimizer
        optimizer = torch.optim.AdamW(
            learning_filter.parameters(), lr=0.0005, weight_decay=0.01
        )

        # Prepare learning process
        learning_process = LearningProcess(
            module=learning_filter,
            loss_function=loss_function,
            optimizer=optimizer,
        )

    else:

        # Load learning process from checkpoint
        learning_process = LearningProcess.from_checkpoint(
            module=learning_filter,
            model_filepath=model_folderpath / model_filename,
            safe_callables={
                torch.nn.functional.cross_entropy,
                torch.optim.AdamW,
                # types of extra arguments
            },
        )

    # Prepare training configuration
    training_config = TrainingConfig()
    training_config.validation.per_iteration_period = 300
    training_config.termination.max_epoch = 16
    training_config.checkpoint.folderpath = (
        model_folderpath / training_config.checkpoint.folderpath
    )
    training_config.checkpoint.filename = model_filename
    training_config.checkpoint.per_epoch_period = 4
    if not new_training:
        training_config.validation.at_initial = False

    # Train model
    with device_manager.monitor_performance(
        sampling_rate=10.0,
        result_directory=result_directory,
    ):
        learning_process.training(
            training_loader, evaluation_loader, training_config
        )

    # Test model
    learning_process.test_model(testing_loader)
