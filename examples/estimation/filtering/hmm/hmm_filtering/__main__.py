import click
import numpy as np

from ss.utility.logging import Logging
from ss.utility.path import PathManager

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
    "--observation-dim",
    type=click.IntRange(min=1),
    default=7,
    help="Set the observation dimension (positive integers).",
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
def main(
    simulation_time_steps: int,
    step_skip: int,
    state_dim: int,
    observation_dim: int,
    number_of_systems: int,
    random_seed: int,
    verbose: bool,
    debug: bool,
) -> None:
    path_manager = PathManager(__file__)
    result_directory = path_manager.result_directory
    Logging.basic_config(
        filename=path_manager.logging_filepath,
        verbose=verbose,
        debug=debug,
    )
    np.random.seed(random_seed)

    hmm_filtering(
        state_dim=state_dim,
        observation_dim=observation_dim,
        simulation_time_steps=simulation_time_steps,
        step_skip=step_skip,
        number_of_systems=number_of_systems,
        result_directory=result_directory,
    )


if __name__ == "__main__":
    main()
