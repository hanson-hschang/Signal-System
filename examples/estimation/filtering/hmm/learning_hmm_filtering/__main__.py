from typing import Optional, assert_never

from pathlib import Path

import click

from ss.learning import Mode
from ss.utility.logging import Logging
from ss.utility.path import PathManager

from . import inference, train, visualization


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
    "--data-foldername",
    type=click.Path(),
    default="hmm_filtering_result",
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
    default="learning_filter.pt",
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
    mode: Mode,
    data_foldername: Path,
    data_filename: Path,
    model_foldername: Optional[Path],
    model_filename: Path,
    verbose: bool,
    debug: bool,
) -> None:
    path_manager = PathManager(__file__)
    Logging.basic_config(
        filename=path_manager.logging_filepath,
        verbose=verbose,
        debug=debug,
    )
    data_filepath = path_manager.get_directory(data_foldername) / data_filename
    model_folderpath = (
        path_manager.result_directory
        if model_foldername is None
        else path_manager.get_directory(model_foldername)
    )
    match mode:
        case Mode.TRAIN:
            model_filepath = (
                model_folderpath / path_manager.current_date / model_filename
                if model_foldername is None
                else model_folderpath / model_filename
            )
            train(data_filepath, model_filepath)
        case Mode.VISUALIZATION:
            model_filepath = model_folderpath / model_filename
            visualization(data_filepath, model_filepath)
        case Mode.INFERENCE:
            model_filepath = model_folderpath / model_filename
            inference(data_filepath, model_filepath)
        case _ as _mode:
            assert_never(_mode)


if __name__ == "__main__":
    main()
