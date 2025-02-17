from typing import Union

import os
from datetime import datetime
from pathlib import Path

from ss.utility.assertion.validator import FolderPathExistenceValidator


class PathManager:
    def __init__(self, file: str) -> None:
        self._file = Path(file)
        self._abspath = os.path.abspath(self._file)
        self._date = datetime.now().strftime(r"%Y%m%d")
        current_directory_path = Path(os.path.dirname(self._abspath))
        self._result_directory_appendix = "_result"
        if self._file.stem == "__main__":
            self._result_directory_path = Path(
                current_directory_path.as_posix()
                + self._result_directory_appendix
            )
            self._current_directory_path = current_directory_path.parent
        else:
            self._result_directory_path = current_directory_path / (
                self._file.stem + self._result_directory_appendix
            )
            self._current_directory_path = current_directory_path
        self._logger_filename = self._date + ".log"

    @property
    def result_directory(self) -> Path:
        return self._result_directory_path

    @result_directory.setter
    def result_directory(self, path: Path) -> None:
        self._result_directory_path = path

    @property
    def current_date(self) -> Path:
        return Path(self._date)

    @property
    def logging_filepath(self) -> Path:
        return self._result_directory_path / Path(self._logger_filename)

    def get_directory(
        self, foldername: Union[str, Path], auto_create: bool = False
    ) -> Path:
        directory = FolderPathExistenceValidator(
            self._current_directory_path / Path(foldername),
            auto_create=auto_create,
        ).get_folderpath()
        return directory
