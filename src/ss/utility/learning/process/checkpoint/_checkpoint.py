from typing import Any, Dict, Optional, Self, Set, Tuple, Union

from pathlib import Path

import h5py
import numpy as np

from ss.utility.assertion.validator import (
    FilePathValidator,
    FolderPathExistenceValidator,
    ReservedKeyNameValidator,
)
from ss.utility.learning import serialization
from ss.utility.learning.module import BLM, BaseLearningModule
from ss.utility.learning.process.checkpoint import config as Config
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class CheckpointInfo(dict):
    FILE_EXTENSION = ".hdf5"
    RESERVED_KEYS_VIEW = dict(__type__="ClassName").keys()

    def __init__(self, **kwargs: Any) -> None:
        ReservedKeyNameValidator(
            kwargs, self.RESERVED_KEYS_VIEW, allow_dunder_names=True
        )
        super().__init__(**kwargs)

    @classmethod
    def load(cls, filename: Union[str, Path]) -> Self:
        filepath = FilePathValidator(
            filename, cls.FILE_EXTENSION
        ).get_filepath()
        with h5py.File(filepath, "r") as f:
            checkpoint_info = cls._load(f)
            for key, value in f.attrs.items():
                checkpoint_info.update({key: value})
        return cls(**checkpoint_info)

    @classmethod
    def _load(cls, group: h5py.Group) -> Dict[str, Any]:
        checkpoint_info: Dict[str, Any] = dict()
        for key, value in group.items():
            if isinstance(value, h5py.Group):
                checkpoint_info[key] = cls._load(value)
            elif isinstance(value, h5py.Dataset):
                match group[key].attrs["__type__"]:
                    case "list":
                        checkpoint_info[key] = list(value)
                    case "tuple":
                        checkpoint_info[key] = tuple(value)
                    case "ndarray":
                        checkpoint_info[key] = np.array(value)
                    case _ as _invalid_type:
                        logger.warning(
                            f"invalid type: {_invalid_type} read from the checkpoint_info file"
                        )
                        logger.warning(f"{key}: {value}")
            else:
                checkpoint_info[key] = value
        return checkpoint_info

    def save(self, filename: Union[str, Path]) -> None:
        filepath = FilePathValidator(
            filename, self.FILE_EXTENSION
        ).get_filepath()
        with h5py.File(filepath, "w") as f:
            ReservedKeyNameValidator(
                self, self.RESERVED_KEYS_VIEW, allow_dunder_names=True
            )
            for key, value in self.items():
                self._save(f, key, value)
        logger.debug(f"checkpoint info saved to {filepath}")

    @classmethod
    def _save(cls, group: h5py.Group, name: str, data: Any) -> None:
        if isinstance(data, dict):
            ReservedKeyNameValidator(
                data, cls.RESERVED_KEYS_VIEW, allow_dunder_names=True
            )
            subgroup = group.create_group(name)
            for key, value in data.items():
                cls._save(subgroup, key, value)
        elif isinstance(data, (list, tuple, np.ndarray)):
            group.create_dataset(name, data=data)
            group[name].attrs["__type__"] = type(data).__name__
        else:
            group.attrs[name] = data


class Checkpoint:
    def __init__(
        self, config: Optional[Config.CheckpointConfig] = None
    ) -> None:
        if config is None:
            config = Config.CheckpointConfig()
        self._config = config
        self._initialize()

    def _initialize(self) -> None:
        self._checkpoint_filepath = (
            FolderPathExistenceValidator(
                foldername=self._config.folderpath,
                auto_create=True,
            ).get_folderpath()
            / self._config.filename
        )
        self._initial_skip = self._config.initial.skip
        self._initial_index = self._config.initial.index
        self._index = self._config.initial.index
        self._finalize = False

    @property
    def config(self) -> Config.CheckpointConfig:
        return self._config

    @config.setter
    def config(self, config: Config.CheckpointConfig) -> None:
        self._config = config
        self._initialize()

    @property
    def checkpoint_appendix(self) -> str:
        return self._config.appendix(self._index)

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
        module: BaseLearningModule,
        checkpoint_info: CheckpointInfo,
        model_info: Dict[str, Any],
    ) -> None:
        if self._index == self._initial_index:
            logger.info(
                f"checkpoints are saved every {self._config.per_epoch_period} epoch(s) "
                + (
                    "with the initial checkpoint skipped"
                    if self._initial_skip
                    else "with the initial checkpoint saved"
                )
            )
            if self._initial_skip:
                self._index += 1
                return
        filepath = self.filepath
        module.save(
            filename=filepath.with_suffix(module.FILE_EXTENSIONS[0]),
            model_info=model_info,
        )
        checkpoint_info.save(
            filename=filepath.with_suffix(CheckpointInfo.FILE_EXTENSION),
        )
        if not self._finalize:
            self._index += 1

    def finalize(self) -> Self:
        self._finalize = True
        return self

    @classmethod
    def load(
        cls,
        module: BLM,
        model_filepath: Path,
        safe_callables: Optional[Set[serialization.SafeCallable]] = None,
    ) -> Tuple[BLM, Dict[str, Any], CheckpointInfo]:
        module_filepath = (
            model_filepath
            if model_filepath.suffix in module.FILE_EXTENSIONS
            else model_filepath.with_suffix(module.FILE_EXTENSIONS[0])
        )
        module, model_info = module.load(
            module_filepath,
            safe_callables,
        )
        checkpoint_info = CheckpointInfo.load(
            model_filepath.with_suffix(CheckpointInfo.FILE_EXTENSION)
        )
        return module, model_info, checkpoint_info
