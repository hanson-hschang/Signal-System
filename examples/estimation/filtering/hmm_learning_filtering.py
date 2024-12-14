from typing import Any, Sequence, Tuple, assert_never

import os
from pathlib import Path

import click
import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split

from lss import Mode
from lss.estimation.filtering.hmm import (
    LearningHiddenMarkovModelFilter,
    LearningHiddenMarkovModelFilterParameters,
)
from ss.utility.data import Data


class ObservationDataset(Dataset):
    def __init__(
        self,
        observation: ArrayLike,
        number_of_systems: int = 1,
        max_length: int = 256,
        stride: int = 64,
    ) -> None:
        if not isinstance(observation, torch.Tensor):
            observation = np.array(observation)
        if number_of_systems == 1:
            observation = observation[np.newaxis, ...]
        assert observation.ndim == 3, (
            "observation must be a 3D tensor with the shape (number_of_systems, observation_dim, time_horizon)."
            f"observation given has the shape of {observation.shape}."
        )

        time_horizon = observation.shape[-1]
        self._input_trajectory = []
        self._output_trajectory = []

        for i in range(number_of_systems):
            for t in range(0, time_horizon - max_length, stride):
                input_trajectory = observation[i, ..., t : t + max_length]
                output_trajectory = observation[
                    i, ..., t + 1 : t + max_length + 1
                ]
                self._input_trajectory.append(torch.tensor(input_trajectory))
                self._output_trajectory.append(torch.tensor(output_trajectory))

        self._length = len(self._input_trajectory)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._input_trajectory[index], self._output_trajectory[index]

    @classmethod
    def from_batch(cls, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        input_trajectory, output_trajectory = batch[0], batch[1]
        return input_trajectory, output_trajectory


def train_test_split(
    observation: ArrayLike,
    split_ratio: Sequence[float],
    number_of_systems: int = 1,
    batch_size: int = 128,
    max_length: int = 256,
    stride: int = 64,
    random_seed: int = 2025,
) -> Tuple[DataLoader, DataLoader]:
    assert len(split_ratio) == 2, (
        "split_ratio must be a list of two floats."
        f"split_ratio given has the length of {len(split_ratio)}."
    )
    generator = torch.Generator().manual_seed(random_seed)
    training_set, testing_set = random_split(
        dataset=ObservationDataset(
            observation=observation,
            number_of_systems=number_of_systems,
            max_length=max_length,
            stride=stride,
        ),
        lengths=split_ratio,
        generator=generator,
    )
    training_loader = DataLoader(
        training_set, batch_size=batch_size, shuffle=True
    )
    testing_loader = DataLoader(
        testing_set, batch_size=batch_size, shuffle=True
    )
    return training_loader, testing_loader


def train(
    data_filename: Path,
    result_directory: Path,
    model_filename: Path,
) -> None:
    # Prepare data
    data = Data.load_from_file(data_filename)
    observation = data["observation"]
    number_of_systems = data.meta_info["number_of_systems"]
    training_loader, testing_loader = train_test_split(
        observation=observation,
        split_ratio=[0.8, 0.2],
        number_of_systems=number_of_systems,
    )

    # Prepare model
    params = LearningHiddenMarkovModelFilterParameters(
        state_dim=5,
        observation_dim=(
            observation.shape[0]
            if number_of_systems == 1
            else observation.shape[1]
        ),
        feature_dim=1,
        layer_dim=1,
    )
    filter = LearningHiddenMarkovModelFilter(params)

    # Prepare loss function
    loss_function = torch.nn.functional.cross_entropy

    # Prepare optimizer
    optimizer = optim.Adam(filter.parameters(), lr=0.001)

    # Train model
    for training_batch in training_loader:
        observation_trajectory, next_observation_trajectory = (
            ObservationDataset.from_batch(training_batch)
        )
        optimizer.zero_grad()
        estimated_next_observation_trajectory = filter(
            observation_trajectory=observation_trajectory
        )
        loss = loss_function(
            estimated_next_observation_trajectory,
            next_observation_trajectory,
        )
        loss.backward()
        optimizer.step()

    # initial_length = 100
    # filter.update(
    #     observation_trajectory=torch.tensor(observation[0, :, :initial_length])
    # )
    # for i in range(5):
    #     filter.update(torch.tensor(observation[0, :, initial_length + i]))
    #     print(filter._estimated_next_state)
    #     filter.estimate()
    #     print(filter.estimated_next_observation)

    # model_filename = result_directory / model_filename
    # filter.save(model_filename)


def inference(
    result_directory: Path,
    model_filename: Path,
) -> None:

    model_filename = result_directory / model_filename
    filter = LearningHiddenMarkovModelFilter.load(model_filename)

    x = torch.tensor([1.0, 2.0])
    for _ in range(5):
        filter.update(observation_trajectory=x)
        filter.estimate()
        print(filter.estimated_next_observation)


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
def main(
    mode: Mode,
    model_filename: Path,
    data_foldername: Path,
    data_filename: Path,
) -> None:
    parent_directory = Path(os.path.dirname(os.path.abspath(__file__)))
    result_directory = parent_directory / Path(__file__).stem
    match mode:
        case Mode.TRAIN:
            data_filename = parent_directory / data_foldername / data_filename
            train(data_filename, result_directory, model_filename)
        case Mode.INFERENCE:
            inference(result_directory, model_filename)
        case _ as _mode:
            assert_never(_mode)


if __name__ == "__main__":
    main()
