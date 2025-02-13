from typing import Any, Dict, Optional, Self, Union

from pathlib import Path

import h5py
import numpy as np

from ss.utility.assertion.validator import (
    FilePathValidator,
    FolderPathExistenceValidator,
)
from ss.utility.learning import module as Module
from ss.utility.learning.process.checkpoint import config as Config
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class CheckpointInfo(dict):
    FILE_EXTENSION = ".hdf5"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @classmethod
    def load(cls, filename: Union[str, Path]) -> Self:
        filepath = FilePathValidator(
            filename, cls.FILE_EXTENSION
        ).get_filepath()
        with h5py.File(filepath, "r") as f:
            checkpoint_info = cls._load(f)
        return cls(**checkpoint_info)

    @classmethod
    def _load(cls, group: h5py.Group) -> Dict[str, Any]:
        checkpoint_info: Dict[str, Any] = dict()
        for key, value in group.items():
            if isinstance(value, h5py.Group):
                checkpoint_info[key] = cls._load(value)
            elif isinstance(value, h5py.Dataset):
                checkpoint_info[key] = np.array(value)
            else:
                checkpoint_info[key] = value
        return checkpoint_info

    def save(self, filename: Union[str, Path]) -> None:
        filepath = FilePathValidator(
            filename, self.FILE_EXTENSION
        ).get_filepath()
        with h5py.File(filepath, "w") as f:
            for key, value in self.items():
                self._save(f, key, value)
        logger.debug(f"checkpoint info saved to {filepath}")

    @classmethod
    def _save(cls, group: h5py.Group, name: str, value: Any) -> None:
        if isinstance(value, dict):
            subgroup = group.create_group(name)
            for key, val in value.items():
                cls._save(subgroup, key, val)
        elif isinstance(value, (list, tuple, np.ndarray)):
            group.create_dataset(name, data=value)
        else:
            group.attrs[name] = value


class Checkpoint:
    def __init__(
        self, config: Optional[Config.CheckpointConfig] = None
    ) -> None:
        if config is None:
            config = Config.CheckpointConfig()
        self._config = config
        self._initialize()
        self._counter = 0
        self._finalize = False

    def _initialize(self) -> None:
        self._checkpoint_filepath = (
            FolderPathExistenceValidator(
                foldername=self._config.filepath.parent,
                auto_create=True,
            ).get_folderpath()
            / self._config.filepath.name
        )

    @property
    def config(self) -> Config.CheckpointConfig:
        return self._config

    @config.setter
    def config(self, config: Config.CheckpointConfig) -> None:
        self._config = config
        self._initialize()

    @property
    def checkpoint_appendix(self) -> str:
        return self._config.appendix(self._counter)

    @property
    def filepath(self) -> Path:
        return (
            self._checkpoint_filepath
            if self._finalize
            else Path(
                f"{self._checkpoint_filepath}" + self.checkpoint_appendix
            )
        )

    def save(
        self,
        module: Module.BaseLearningModule,
        checkpoint_info: CheckpointInfo,
    ) -> None:
        if self._counter == 0:
            logger.info(
                f"checkpoints are saved every {self._config.per_epoch_period} epoch(s)"
            )
        filepath = self.filepath
        module.save(
            filename=filepath.with_suffix(
                Module.BaseLearningModule.FILE_EXTENSIONS[0]
            ),
        )
        checkpoint_info.save(
            filename=filepath.with_suffix(CheckpointInfo.FILE_EXTENSION),
        )
        if not self._finalize:
            self._counter += 1

    def finalize(self) -> Self:
        self._finalize = True
        return self
