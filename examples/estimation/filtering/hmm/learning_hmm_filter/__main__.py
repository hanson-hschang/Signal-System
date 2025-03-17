from typing import Optional, assert_never

from pathlib import Path

import click

from ss.utility import basic_config
from ss.utility.learning.process import BaseLearningProcess

from . import analysis, inference, training


@click.command()
@click.option(
    "--mode",
    type=click.Choice(
        [mode for mode in BaseLearningProcess.Mode],
        case_sensitive=False,
    ),
    default=BaseLearningProcess.Mode.INFERENCE,
)
@click.option(
    "--data-foldername",
    type=click.Path(),
    default="hmm_filter_result",
)
@click.option(
    "--data-filename",
    type=click.Path(),
    default="system_train.hdf5",
)
@click.option(
    "--model-foldername",
    type=click.Path(),
    default=None,
)
@click.option(
    "--model-filename",
    type=click.Path(),
    default="learning_filter",
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
@click.option(
    "--continue-training",
    is_flag=True,
    help="Continue training the model.",
)
def main(
    mode: BaseLearningProcess.Mode,
    data_foldername: str,
    data_filename: str,
    model_foldername: Optional[str],
    model_filename: str,
    verbose: bool,
    debug: bool,
    result_directory: Optional[Path],
    continue_training: bool,
) -> None:
    path_manager = basic_config(
        __file__,
        verbose=verbose,
        debug=debug,
        result_directory=result_directory,
    )
    data_filepath = path_manager.get_directory(data_foldername) / data_filename
    model_folderpath = (
        path_manager.result_directory
        if model_foldername is None
        else path_manager.get_directory(model_foldername)
    )
    result_directory = path_manager.result_directory
    match mode:
        case BaseLearningProcess.Mode.TRAINING:
            # model_filepath = model_folderpath / "checkpoints" / model_filename
            training(
                data_filepath,
                model_folderpath,
                model_filename,
                result_directory,
                not continue_training,
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
