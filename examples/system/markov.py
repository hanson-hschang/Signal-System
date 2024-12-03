import os
from pathlib import Path

import click
from tqdm import tqdm

from ss.system.markov import HiddenMarkovModel, MarkovChainCallback


@click.command()
@click.option(
    "--simulation-time-steps",
    type=click.IntRange(min=1),
    default=100,
    help="Set the simulation time steps (positive integers).",
)
@click.option(
    "--step-skip",
    type=click.IntRange(min=1),
    default=1,
    help="Set the step skip (positive integers).",
)
@click.option(
    "--number-of-systems",
    type=click.IntRange(min=1),
    default=1,
    help="Set the number of systems (positive integers).",
)
def main(
    simulation_time_steps: int,
    step_skip: int,
    number_of_systems: int,
) -> None:
    epsilon = 0.01
    transition_probability_matrix = [
        [0, 0.5, 0.5],
        [epsilon, 1 - epsilon, 0],
        [1 - epsilon, 0, epsilon],
    ]
    emission_probability_matrix = [
        [1, 0],
        [0, 1],
        [0, 1],
    ]

    markov_chain = HiddenMarkovModel(
        transition_probability_matrix=transition_probability_matrix,
        emission_probability_matrix=emission_probability_matrix,
        number_of_systems=number_of_systems,
    )
    system_callback = MarkovChainCallback(
        step_skip=step_skip,
        system=markov_chain,
    )

    current_time = 0.0
    for k in tqdm(range(simulation_time_steps)):
        system_callback.record(k, current_time)
        current_time = markov_chain.process(current_time)

    system_callback.record(simulation_time_steps, current_time)

    # Save the data
    parent_directory = Path(os.path.dirname(os.path.abspath(__file__)))
    data_folder_directory = parent_directory / Path(__file__).stem
    system_callback.save(data_folder_directory / "system.hdf5")


if __name__ == "__main__":
    main()
