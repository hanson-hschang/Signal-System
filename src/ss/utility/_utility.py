from typing import Optional

from pathlib import Path

from ss.utility.logging import Logging
from ss.utility.path import PathManager


def basic_config(
    file: str,
    verbose: bool,
    debug: bool,
    logging_filename: Optional[str] = None,
) -> Path:
    path_manager = PathManager(file)
    Logging.basic_config(
        filename=(
            logging_filename
            if logging_filename
            else path_manager.logging_filepath
        ),
        verbose=verbose,
        debug=debug,
    )
    result_directory = path_manager.result_directory
    return result_directory
