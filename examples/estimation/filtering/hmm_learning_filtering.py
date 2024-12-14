from typing import assert_never

import os
from pathlib import Path

import click
import torch
from torch import optim

from lss import Mode
from lss.estimation.filtering.hmm import (
    LearningHiddenMarkovModelFilter,
    LearningHiddenMarkovModelFilterParameters,
)
from ss.utility.data import Data


def train(
    model_filename: Path,
    data_filename: Path,
) -> None:

    data = Data.load_from_file(data_filename)
    observation = data["observation"]

    params = LearningHiddenMarkovModelFilterParameters(
        state_dim=5,
        observation_dim=(
            observation.shape[0]
            if data.meta_info["number_of_systems"] == 1
            else observation.shape[1]
        ),
        feature_dim=1,
        layer_dim=1,
    )
    filter = LearningHiddenMarkovModelFilter(params)
    optimizer = optim.Adam(filter.parameters(), lr=0.001)

    initial_length = 100
    filter.update(
        observation_trajectory=torch.tensor(observation[0, :, :initial_length])
    )
    for i in range(5):
        filter.update(torch.tensor(observation[0, :, initial_length + i]))
        print(filter._estimated_next_state)
        filter.estimate()
        print(filter.estimated_next_observation)

    # filter.save(model_filename)


def inference(
    model_filename: Path,
) -> None:

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
    result_folder_directory = parent_directory / Path(__file__).stem
    model_filename = result_folder_directory / model_filename
    match mode:
        case Mode.TRAIN:
            data_filename = parent_directory / data_foldername / data_filename
            train(model_filename, data_filename)
        case Mode.INFERENCE:
            inference(model_filename)
        case _ as _mode:
            assert_never(_mode)


if __name__ == "__main__":
    main()
