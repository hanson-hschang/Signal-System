from typing import Optional, Union

from pathlib import Path

from ss.utility.logging import Logging
from ss.utility.path import PathManager


def basic_config(
    file: str,
    verbose: bool,
    debug: bool,
    result_directory: Optional[Union[str, Path]] = None,
) -> Path:
    path_manager = PathManager(file)
    result_directory = (
        Path(result_directory)
        if result_directory
        else path_manager.result_directory
    )
    Logging.basic_config(
        filename=(
            result_directory / path_manager.current_date.with_suffix(".log")
            if result_directory
            else path_manager.logging_filepath
        ),
        verbose=verbose,
        debug=debug,
    )
    return result_directory
