from typing import Any, Optional, assert_never

from pathlib import Path

import click

from ss.utility import basic_config
from ss.utility.learning.process import BaseLearningProcess

from . import ClickConfig, analysis, inference, training


@click.command()
@ClickConfig.options(allow_file_overwrite=True)
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

    click_config = ClickConfig.load(config_filepath, **kwargs)

    path_manager = basic_config(
        __file__,
        verbose=verbose,
        debug=debug,
        result_directory=result_directory,
    )

    data_filepath = (
        path_manager.get_directory(click_config.data_foldername)
        / click_config.data_filename
    )
    model_folderpath = (
        path_manager.result_directory
        if click_config.model_foldername is None
        else path_manager.get_directory(click_config.model_foldername)
    )
    if result_directory is None:
        result_directory = path_manager.result_directory
    model_filename = click_config.model_filename
    match click_config.mode:
        case BaseLearningProcess.Mode.TRAINING:
            # model_filepath = model_folderpath / "checkpoints" / model_filename
            training(
                data_filepath,
                model_folderpath,
                model_filename,
                result_directory,
                not click_config.continue_training,
            )
        case BaseLearningProcess.Mode.ANALYSIS:
            # model_filepath = model_folderpath / model_filename
            analysis(
                data_filepath,
                model_folderpath,
                model_filename,
                result_directory,
            )
        case BaseLearningProcess.Mode.INFERENCE:
            # model_filepath = model_folderpath / model_filename
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
