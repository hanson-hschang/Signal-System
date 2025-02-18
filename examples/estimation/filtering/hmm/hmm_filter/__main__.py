from pathlib import Path

import click
import numpy as np

from ss.utility import basic_config

from . import hmm_filtering


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
    "--state-dim",
    type=click.IntRange(min=1),
    default=3,
    help="Set the state dimension (positive integers).",
)
@click.option(
    "--discrete-observation-dim",
    type=click.IntRange(min=1),
    default=7,
    help="Set the discrete observation dimension (positive integers).",
)
@click.option(
    "--number-of-systems",
    type=click.IntRange(min=1),
    default=1,
    help="Set the number of systems (positive integers).",
)
@click.option(
    "--random-seed",
    type=click.IntRange(min=0),
    default=2024,
    help="Set the random seed (non-negative integers).",
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
@click.option(
    "--result-directory",
    type=click.Path(),
    default=None,
)
def main(
    simulation_time_steps: int,
    step_skip: int,
    state_dim: int,
    discrete_observation_dim: int,
    number_of_systems: int,
    random_seed: int,
    verbose: bool,
    debug: bool,
    result_directory: Path,
) -> None:
    path_manager = basic_config(
        __file__,
        verbose=verbose,
        debug=debug,
        result_directory=result_directory,
    )
    np.random.seed(random_seed)

    hmm_filtering(
        state_dim=state_dim,
        discrete_observation_dim=discrete_observation_dim,
        simulation_time_steps=simulation_time_steps,
        step_skip=step_skip,
        number_of_systems=number_of_systems,
        result_directory=path_manager.result_directory,
    )


if __name__ == "__main__":
    main()
