from typing import Any, Optional, no_type_check

from pathlib import Path

import click
import numpy as np

from ss.utility import basic_config

from . import UserConfig, hmm_filtering


@no_type_check
@click.command()
@UserConfig.options(allow_file_overwrite=True)
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
    config_filepath: Optional[Path],
    verbose: bool,
    debug: bool,
    result_directory: Path,
    **kwargs: Any,
) -> None:

    user_config = UserConfig.load(config_filepath, **kwargs)

    path_manager = basic_config(
        __file__,
        verbose=verbose,
        debug=debug,
        result_directory=result_directory,
    )
    np.random.seed(user_config.random_seed)

    hmm_filtering(
        state_dim=user_config.state_dim,
        discrete_observation_dim=user_config.discrete_observation_dim,
        simulation_time_steps=user_config.simulation_time_steps,
        step_skip=user_config.step_skip,
        batch_size=user_config.batch_size,
        result_directory=path_manager.result_directory,
    )


if __name__ == "__main__":
    main()
