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
        self._parent_directory_path = Path(os.path.dirname(self._abspath))
        self._result_directory_path = (
            self._parent_directory_path / self._file.stem
        )

    @property
    def parent_directory(self) -> Path:
        return self._parent_directory_path

    @property
    def result_directory(self) -> Path:
        return self._result_directory_path / Path(self._date)

    def get_other_result_directory(self, foldername: Union[str, Path]) -> Path:
        other_result_directory = FolderPathExistenceValidator(
            self._result_directory_path / Path(foldername)
        ).get_folderpath()
        return other_result_directory
