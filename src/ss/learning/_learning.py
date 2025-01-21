from types import TracebackType
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generic,
    List,
    Optional,
    Self,
    Type,
    TypeVar,
    Union,
    cast,
)

from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ss.utility.assertion.validator import (
    FilePathValidator,
    NonnegativeIntegerValidator,
    PositiveIntegerValidator,
)
from ss.utility.learning.registration import (
    register_numpy,
    register_subclasses,
)
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


BLC = TypeVar("BLC", bound="BaseLearningConfig")


@dataclass
class BaseLearningConfig:
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert dataclass to a dictionary suitable for ** unpacking.
        Uses dataclasses.asdict() for nested conversion.
        """
        return asdict(self)


BLM = TypeVar("BLM", bound="BaseLearningModule")


class BaseLearningModule(nn.Module, Generic[BLC]):
    MODEL_FILE_EXTENSION = (".pt", ".pth")

    def __init__(self, config: Optional[BLC] = None) -> None:
        super().__init__()
        if config is None:
            self._config: BLC = cast(BLC, BaseLearningConfig())
        else:
            assert issubclass(
                type(config), BaseLearningConfig
            ), f"{type(config) = } must be a subclass of {BaseLearningConfig}"
            self._config = config
        self.inference = False

    def reset(self) -> None: ...

    def inference_mode(self, inference: bool = True) -> None:
        if inference:
            self.eval()
        else:
            self.train()
        self._inference_mode(inference)

    def _inference_mode(self, inference: bool) -> None:
        self.inference = inference
        for member in vars(self).values():
            if isinstance(member, BaseLearningModule):
                member._inference_mode(inference)

    def save(
        self,
        filename: Union[str, Path],
    ) -> None:
        filepath = FilePathValidator(
            filename, BaseLearningModule.MODEL_FILE_EXTENSION
        ).get_filepath()
        model_info = dict(
            config=self._config,
            model_state_dict=self.state_dict(),
        )
        torch.save(model_info, filepath)

    @classmethod
    def load(cls: Type[BLM], filename: Union[str, Path]) -> BLM:
        filepath = FilePathValidator(
            filename, BaseLearningModule.MODEL_FILE_EXTENSION
        ).get_filepath()

        register_subclasses(BaseLearningConfig, "ss")
        # register_numpy() # Uncomment this line to register numpy types
        model_info: Dict[str, Any] = torch.load(filepath, weights_only=True)
        config = model_info.pop("config")
        config_dict: Dict[str, Any] = config.__dict__
        config_field = {
            key: value
            for key, value in config_dict.items()
            if not key.startswith("_")
        }
        model = cls(config.__class__(**config_field))
        model.load_state_dict(model_info.pop("model_state_dict"))
        return model


def reset_module(instance: Any) -> None:
    reset_method: Optional[Callable[[], Any]] = getattr(
        instance, "reset", None
    )
    if callable(reset_method):
        reset_method()


class CheckpointInfo(dict):
    _data_file_extension = ".hdf5"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @classmethod
    def load(cls, filename: Union[str, Path]) -> Self:
        filepath = FilePathValidator(
            filename, cls._data_file_extension
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
            filename, self._data_file_extension
        ).get_filepath()
        with h5py.File(filepath, "w") as f:
            for key, value in self.items():
                self._save(f, key, value)

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


class BaseLearningProcess:
    class _NumberOfEpochsValidator(PositiveIntegerValidator):
        def __init__(self, number_of_epochs: int) -> None:
            super().__init__(number_of_epochs, "number_of_epochs")

    class _SaveModelEpochSkipValidator(NonnegativeIntegerValidator):
        def __init__(self, save_model_epoch_skip: int) -> None:
            super().__init__(save_model_epoch_skip, "save_model_epoch_skip")

    def __init__(
        self,
        model: BaseLearningModule,
        loss_function: Callable[..., torch.Tensor],
        optimizer: torch.optim.Optimizer,
        number_of_epochs: int,
        model_filename: Union[str, Path],
        save_model_epoch_skip: int = 0,
    ) -> None:
        self._model = model
        self._loss_function = loss_function
        self._optimizer = optimizer
        self._number_of_epochs = self._NumberOfEpochsValidator(
            number_of_epochs
        ).get_value()
        self._model_filepath = FilePathValidator(
            model_filename,
            BaseLearningModule.MODEL_FILE_EXTENSION,
        ).get_filepath()
        self._save_model_step_skip = self._SaveModelEpochSkipValidator(
            save_model_epoch_skip
        ).get_value()

        self._iteration_idx = 0
        self._epoch_history: DefaultDict[str, List[int]] = defaultdict(list)
        self._training_loss = 0.0
        self._evaluation_loss_history: DefaultDict[str, List[float]] = (
            defaultdict(list)
        )
        self._training_loss_history: DefaultDict[str, List[float]] = (
            defaultdict(list)
        )

        self._save_intermediate_models = self._save_model_step_skip > 0
        if self._save_intermediate_models:
            self._digits_of_number_of_checkpoints = (
                len(str(self._number_of_epochs // self._save_model_step_skip))
                + 1
            )
            self._intermediate_model_filename = (
                str(self._model_filepath.with_suffix(""))
                + "_checkpoint_{:0"
                + str(self._digits_of_number_of_checkpoints)
                + "d}"
                + self._model_filepath.suffix
            )

    def update_epoch(self, epoch_idx: int) -> None:
        self._epoch_history["iteration"].append(self._iteration_idx)
        self._epoch_history["epoch"].append(epoch_idx)
        logger.info(f"Finish epoch: {epoch_idx} / {self._number_of_epochs}")
        logger.info("")

    def update_evaluation_loss(self, loss: float) -> None:
        self._evaluation_loss_history["iteration"].append(self._iteration_idx)
        self._evaluation_loss_history["loss"].append(loss)
        logger.info(f"Evaluation loss: {loss}")

    def update_training_loss(self, loss: float) -> None:
        self._training_loss_history["iteration"].append(self._iteration_idx)
        self._training_loss_history["loss"].append(loss)

    def _evaluate_one_batch(self, data_batch: Any) -> torch.Tensor:
        raise NotImplementedError

    def evaluate_model(self, data_loader: DataLoader) -> float:
        logger.info("Evaluating model...")
        self._model.eval()
        loss = 0.0
        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                loss += self._evaluate_one_batch(data_batch).item()
        loss /= i + 1
        return loss

    def _train_one_batch(self, data_batch: Any) -> torch.Tensor:
        self._optimizer.zero_grad()
        _loss = self._evaluate_one_batch(data_batch)
        _loss.backward()
        self._optimizer.step()
        return _loss

    def train_model(self, data_loader: DataLoader) -> float:
        logger.info("Training one epoch...")
        self._model.train()
        with logging_redirect_tqdm(loggers=[logger]):
            for data_batch in tqdm(data_loader, total=len(data_loader)):
                loss = self._train_one_batch(data_batch).item()
                self.update_training_loss(loss)
                self._training_loss = (loss + self._training_loss) / 2
                self._iteration_idx += 1
        return self._training_loss

    def train(
        self,
        training_loader: DataLoader,
        evaluation_loader: DataLoader,
    ) -> None:

        logger.info("Model evaluation before training...")
        epoch_idx, checkpoint_idx = 0, 0

        loss = self.evaluate_model(evaluation_loader)
        self.update_evaluation_loss(loss)

        self.update_epoch(epoch_idx)
        checkpoint_idx = self.save_intermediate_model(
            epoch_idx, checkpoint_idx
        )

        logger.info("Start training...")
        for epoch_idx in range(1, self._number_of_epochs + 1):

            loss = self.train_model(training_loader)
            logger.info(f"Training loss (running average): {loss}")

            loss = self.evaluate_model(evaluation_loader)
            self.update_evaluation_loss(loss)

            self.update_epoch(epoch_idx)
            checkpoint_idx = self.save_intermediate_model(
                epoch_idx, checkpoint_idx
            )

        logger.info("Training is completed")
        self.save_model()

    def test_model(
        self,
        testing_loader: DataLoader,
    ) -> None:
        logger.info("Testing...")
        loss = self.evaluate_model(testing_loader)
        logger.info(f"Testing is completed with loss: {loss}")

    def save_intermediate_model(
        self, epoch_idx: int, checkpoint_idx: int
    ) -> int:
        if self._save_intermediate_models:
            if epoch_idx == 0:
                logger.info(
                    f"Intermediate models are saved every {self._save_model_step_skip} epoch(s)"
                )
            if epoch_idx % self._save_model_step_skip == 0:
                self.save_model(checkpoint_idx)
                return checkpoint_idx + 1
        return checkpoint_idx

    def save_model(self, checkpoint_idx: Optional[int] = None) -> None:
        if checkpoint_idx is None:
            model_filepath = self._model_filepath
        else:
            model_filepath = Path(
                self._intermediate_model_filename.format(checkpoint_idx)
            )
        checkpoint_filepath = model_filepath.with_suffix(".hdf5")

        self._model.save(
            filename=model_filepath,
        )
        logger.debug(f"model saved to {model_filepath}")

        checkpoint_info = self.create_checkpoint_info()
        checkpoint_info.save(checkpoint_filepath)
        logger.debug(f"checkpoint info saved to {checkpoint_filepath}")

    def create_checkpoint_info(self) -> CheckpointInfo:
        checkpoint_info = CheckpointInfo(
            epoch_history=self._epoch_history,
            evaluation_loss_history=self._evaluation_loss_history,
            training_loss_history=self._training_loss_history,
        )
        return checkpoint_info


class InferenceContext:
    def __init__(self, *modules: BaseLearningModule) -> None:
        self._modules = modules
        self._previous_modes: List[bool] = [False] * len(modules)
        self._no_grad = torch.no_grad()

    def __enter__(self) -> None:
        self._no_grad.__enter__()

        for i, module in enumerate(self._modules):
            self._previous_modes[i] = module.inference
            module.inference_mode(True)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        for module, inference in zip(self._modules, self._previous_modes):
            module.inference_mode(inference)

        self._no_grad.__exit__(exc_type, exc_value, traceback)


class Mode(StrEnum):
    TRAIN = "TRAIN"
    VISUALIZATION = "VISUALIZATION"
    INFERENCE = "INFERENCE"

    @classmethod
    def inference(cls, *modules: BaseLearningModule) -> InferenceContext:
        return InferenceContext(*modules)
