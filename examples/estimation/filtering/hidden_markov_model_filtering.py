import os
from pathlib import Path

import click
from tqdm import tqdm

from ss.estimation.estimator import EstimatorCallback
from ss.estimation.filtering.hmm_filtering import HiddenMarkovModelFilter
from ss.system.finite_state.markov import HiddenMarkovModel, MarkovChainCallback


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

    system = HiddenMarkovModel(
        transition_probability_matrix=[
            [0, 0.5, 0.5],
            [epsilon, 1 - epsilon, 0],
            [1 - epsilon, 0, epsilon],
        ],
        emission_probability_matrix=[
            [1, 0],
            [0, 1],
            [0, 1],
        ],
        number_of_systems=number_of_systems,
    )
    system_callback = MarkovChainCallback(step_skip=step_skip, system=system)
    estimator = HiddenMarkovModelFilter(
        system=system,
    )
    estimator_callback = EstimatorCallback(
        step_skip=step_skip,
        estimator=estimator,
    )

    current_time = 0.0
    for k in tqdm(range(simulation_time_steps)):
        system_callback.record(k, current_time)
        estimator_callback.record(k, current_time)

        current_time = system.process(current_time)

        estimator.update(
            observation=system.observe(),
        )
        estimator.estimate()

    system_callback.record(simulation_time_steps, current_time)
    estimator_callback.record(simulation_time_steps, current_time)

    # Save the data
    parent_directory = Path(os.path.dirname(os.path.abspath(__file__)))
    data_folder_directory = parent_directory / Path(__file__).stem
    system_callback.save(data_folder_directory / "system.hdf5")
    estimator_callback.save(data_folder_directory / "filter.hdf5")


if __name__ == "__main__":
    main()
