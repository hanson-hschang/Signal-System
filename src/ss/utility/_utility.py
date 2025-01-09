from pathlib import Path

from ss.utility.logging import Logging
from ss.utility.path import PathManager


def basic_config(
    file: str,
    verbose: bool,
    debug: bool,
) -> Path:
    path_manager = PathManager(file)
    Logging.basic_config(
        filename=path_manager.logging_filepath,
        verbose=verbose,
        debug=debug,
    )
    result_directory = path_manager.result_directory
    return result_directory
