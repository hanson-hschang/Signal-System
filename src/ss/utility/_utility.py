from pathlib import Path

from ss.utility.logging import Logging
from ss.utility.path import PathManager


def basic_config(
    file: str,
    verbose: bool,
    debug: bool,
    result_directory: str | Path | None = None,
) -> PathManager:
    path_manager = PathManager(file)
    if result_directory:
        path_manager.result_directory = Path(result_directory)
    Logging.basic_config(
        filename=path_manager.logging_filepath,
        verbose=verbose,
        debug=debug,
    )
    return path_manager
