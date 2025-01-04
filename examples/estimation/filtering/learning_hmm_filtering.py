from typing import Any, List, Sequence, Tuple, assert_never

import os
from pathlib import Path

import click
import torch
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from torch.utils.data import DataLoader, Dataset, random_split

from ss.estimation.filtering.hmm_filtering import (
    LearningHiddenMarkovModelFilter,
    LearningHiddenMarkovModelFilterParameters,
)
from ss.learning import (
    BaseLearningProcess,
    CheckpointInfo,
    IterationFigure,
    Mode,
)
from ss.utility.data import Data
from ss.utility.logging import Logging
from ss.utility.path import PathManager

logger = Logging.get_logger(__name__)


class ObservationDataset(Dataset):
    def __init__(
        self,
        observation: ArrayLike,
        number_of_systems: int = 1,
        max_length: int = 256,
        stride: int = 64,
    ) -> None:
        observation = torch.tensor(observation)
        if number_of_systems == 1:
            observation = observation.unsqueeze(0)
        assert observation.ndim >= 3, (
            "observation must be a tensor greater than 3 dimensions "
            "with the shape of (number_of_systems, ..., time_horizon). "
            f"observation given has the shape of {observation.shape}."
        )

        time_horizon = observation.shape[-1]
        self._input_trajectory = []
        self._output_trajectory = []

        with torch.no_grad():
            for i in range(number_of_systems):
                for t in range(0, time_horizon - max_length, stride):
                    input_trajectory = observation[
                        i, ..., t : t + max_length
                    ]  # (observation_dim, max_length)
                    output_trajectory = observation[
                        i, ..., t + 1 : t + max_length + 1
                    ]  # (observation_dim, max_length)
                    self._input_trajectory.append(
                        torch.moveaxis(input_trajectory, 1, 0)
                        .detach()
                        .clone()  # (max_length, observation_dim)
                    )
                    self._output_trajectory.append(
                        torch.moveaxis(output_trajectory, 1, 0)
                        .detach()
                        .clone()  # (max_length, observation_dim)
                    )

        self._length = len(self._input_trajectory)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._input_trajectory[index], self._output_trajectory[index]

    @classmethod
    def from_batch(cls, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        # The indexing of batch is defined by the __getitem__ method.
        input_trajectory, output_trajectory = batch[0], batch[1]
        return input_trajectory, output_trajectory


def data_split(
    observation: ArrayLike,
    split_ratio: Sequence[float],
    number_of_systems: int = 1,
    batch_size: int = 128,
    max_length: int = 256,
    stride: int = 64,
    random_seed: int = 2025,
) -> List[DataLoader]:
    generator = torch.Generator().manual_seed(random_seed)
    datasets = random_split(
        dataset=ObservationDataset(
            observation=observation,
            number_of_systems=number_of_systems,
            max_length=max_length,
            stride=stride,
        ),
        lengths=split_ratio,
        generator=generator,
    )
    dataloaders = []
    for dataset in datasets:
        dataloaders.append(
            DataLoader(dataset, batch_size=batch_size, shuffle=True)
        )
    return dataloaders


class LearningHMMFilterProcess(BaseLearningProcess):
    def _train_one_batch(self, data_batch: Any) -> float:
        observation_trajectory, next_observation_trajectory = (
            ObservationDataset.from_batch(data_batch)
        )
        self._optimizer.zero_grad()
        estimated_next_observation_trajectory = self._model(
            observation_trajectory=observation_trajectory
        )
        loss = self._loss_function(
            torch.moveaxis(estimated_next_observation_trajectory, 1, 2),
            torch.moveaxis(next_observation_trajectory, 1, 2),
        )
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def _evaluate_one_batch(self, data_batch: Any) -> float:
        observation_trajectory, next_observation_trajectory = (
            ObservationDataset.from_batch(data_batch)
        )
        estimated_next_observation_trajectory = self._model(
            observation_trajectory=observation_trajectory
        )
        loss = self._loss_function(
            torch.moveaxis(estimated_next_observation_trajectory, 1, 2),
            torch.moveaxis(next_observation_trajectory, 1, 2),
        )
        return loss.item()


def train(
    data_filename: Path,
    result_directory: Path,
    model_filename: Path,
) -> None:
    # Prepare data
    data = Data.load(data_filename)
    observation = data["observation"]
    number_of_systems = data.meta_info["number_of_systems"]
    observation_dim = (
        observation.shape[0] if number_of_systems == 1 else observation.shape[1]
    )
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
    )
    filter = LearningHiddenMarkovModelFilter(params)

    # Prepare loss function
    loss_function = torch.nn.functional.cross_entropy

    # Prepare optimizer
    optimizer = torch.optim.AdamW(filter.parameters(), lr=0.001)

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
    learning_process.test(testing_loader)


def visualization(
    result_directory: Path,
    model_filename: Path,
) -> None:
    model_filename = result_directory / model_filename
    filter = LearningHiddenMarkovModelFilter.load(model_filename)
    logger.info(filter.emission_matrix)
    logger.info(
        torch.nn.functional.softmax(
            filter._transition_layer.layers[0].blocks[0]._weight,
            dim=1,
        )
    )

    checkpoint_info = CheckpointInfo.load(model_filename.with_suffix(".hdf5"))
    IterationFigure(
        training_loss_trajectory=checkpoint_info["training_loss_history"],
        validation_loss_trajectory=checkpoint_info["evaluation_loss_history"],
    ).plot()
    plt.show()


def inference(
    result_directory: Path,
    model_filename: Path,
) -> None:

    model_filename = result_directory / model_filename
    filter = LearningHiddenMarkovModelFilter.load(model_filename)
    logger.info(filter.emission_matrix)
    transition_probability_matrix = torch.nn.functional.softmax(
        filter._transition_layer.layers[0].blocks[0]._weight,
        dim=1,
    )
    logger.info(transition_probability_matrix)

    observation_trajectory = torch.tensor(
        [
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [0, 1],
            [1, 0],
            [0, 1],
        ]
    )
    filter.update(observation_trajectory)
    for _ in range(5):
        filter.estimate()
        estimated_next_observation = filter.estimated_next_observation
        logger.info(estimated_next_observation)
        filter.update(estimated_next_observation)


@click.command()
@click.option(
    "--mode",
    type=click.Choice(
        [mode for mode in Mode],
        case_sensitive=False,
    ),
    default=Mode.INFERENCE,
)
@click.option(
    "--model-filename",
    type=click.Path(),
    default="learning_filter.pt",
)
@click.option(
    "--data-foldername",
    type=click.Path(),
    default="hmm_filtering",
)
@click.option(
    "--data-filename",
    type=click.Path(),
    default="system.hdf5",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Set the verbose mode.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Set the debug mode.",
)
def main(
    mode: Mode,
    model_filename: Path,
    data_foldername: Path,
    data_filename: Path,
    verbose: bool,
    debug: bool,
) -> None:
    # FIXME: the --verbose and --debug options are not working

    path_manager = PathManager(__file__)
    parent_directory = path_manager.parent_directory
    result_directory = path_manager.result_directory
    Logging.basic_config(
        filename=path_manager.logging_filepath,
        log_level=Logging.Level.DEBUG if debug else Logging.Level.INFO,
        verbose_level=Logging.Level.INFO if verbose else Logging.Level.WARNING,
    )

    match mode:
        case Mode.TRAIN:
            data_filename = parent_directory / data_foldername / data_filename
            train(
                data_filename,
                result_directory / path_manager.current_date_directory,
                model_filename,
            )
        case Mode.VISUALIZATION:
            visualization(result_directory, model_filename)
        case Mode.INFERENCE:
            inference(result_directory, model_filename)
        case _ as _mode:
            assert_never(_mode)


if __name__ == "__main__":
    main()
