from typing import assert_never

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
    "--model-filename",
    type=click.Path(),
    default="learning_filter.pt",
)
@click.option(
    "--data-foldername",
    type=click.Path(),
    default="../hmm_filtering_result",
)
@click.option(
    "--data-filename",
    type=click.Path(),
    default="system.hdf5",
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
    model_filename: Path,
    data_foldername: Path,
    data_filename: Path,
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

    match mode:
        case Mode.TRAIN:
            data_filename = (
                path_manager.get_other_result_directory(
                    path_manager.parent_directory / data_foldername
                )
                / data_filename
            )
            train(
                data_filename,
                result_directory / path_manager.current_date_directory,
                model_filename,
            )
        case Mode.VISUALIZATION:
            data_filename = (
                path_manager.parent_directory / data_foldername / data_filename
            )
            visualization(data_filename, result_directory, model_filename)
        case Mode.INFERENCE:
            inference(result_directory, model_filename)
        case _ as _mode:
            assert_never(_mode)


if __name__ == "__main__":
    main()
