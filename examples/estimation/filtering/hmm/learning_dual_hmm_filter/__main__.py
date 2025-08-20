from typing import Any, Optional, assert_never, no_type_check

from pathlib import Path

import click

from ss.utility import basic_config
from ss.utility.learning.process.config import ProcessConfig

from . import UserConfig, analysis, inference, training


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
    result_directory: Optional[Path],
    **kwargs: Any,
) -> None:

    user_config = UserConfig.load(config_filepath, **kwargs)

    path_manager = basic_config(
        __file__,
        verbose=verbose,
        debug=debug,
        result_directory=result_directory,
    )

    data_filepath = (
        path_manager.get_directory(user_config.data_foldername)
        / user_config.data_filename
    )
    model_folderpath = (
        path_manager.result_directory
        if user_config.model_foldername is None
        else path_manager.get_directory(user_config.model_foldername)
    )
    if result_directory is None:
        result_directory = path_manager.result_directory
    model_filename = user_config.model_filename
    match user_config.mode:
        case ProcessConfig.Mode.TRAINING:
            training(
                data_filepath,
                model_folderpath,
                model_filename,
                result_directory,
                not user_config.continue_training,
            )
        case ProcessConfig.Mode.ANALYSIS:
            analysis(
                data_filepath,
                model_folderpath,
                model_filename,
                result_directory,
            )
        case ProcessConfig.Mode.INFERENCE:
            inference(
                data_filepath,
                model_folderpath,
                model_filename,
                result_directory,
            )
        case _ as _mode:
            assert_never(_mode)


if __name__ == "__main__":
    main()
