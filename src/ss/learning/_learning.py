from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    Self,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from collections import defaultdict
from dataclasses import asdict, dataclass
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
from ss.utility.learning.registration import register_numpy, register_subclasses
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class Mode(StrEnum):
    TRAIN = "TRAIN"
    VISUALIZATION = "VISUALIZATION"
    INFERENCE = "INFERENCE"


@dataclass
class BaseLearningParameters:

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert dataclass to a dictionary suitable for ** unpacking.
        Uses dataclasses.asdict() for nested conversion.
        """
        return asdict(self)


# @dataclass
# class LearningProcessInfo(BaseLearningParameters):
#     epoch: int = 0
#     number_of_epochs: int = 0


BLM = TypeVar("BLM", bound="BaseLearningModule")


class BaseLearningModule(nn.Module):

    MODEL_FILE_EXTENSION = (".pt", ".pth")

    def __init__(self, params: BaseLearningParameters) -> None:
        super().__init__()
        assert issubclass(
            type(params), BaseLearningParameters
        ), f"{type(params) = } must be a subclass of {BaseLearningParameters}"
        self._params = params

    def save(
        self,
        filename: Union[str, Path],
    ) -> None:
        filepath = FilePathValidator(
            filename, BaseLearningModule.MODEL_FILE_EXTENSION
        ).get_filepath()
        model_info = dict(
            params=self._params,
            model_state_dict=self.state_dict(),
        )
        torch.save(model_info, filepath)

    @classmethod
    def load(cls: Type[BLM], filename: Union[str, Path]) -> BLM:
        filepath = FilePathValidator(
            filename, BaseLearningModule.MODEL_FILE_EXTENSION
        ).get_filepath()

        register_subclasses(BaseLearningParameters, "ss")
        register_numpy()
        model_info: Dict[str, Any] = torch.load(filepath, weights_only=True)
        model = cls(model_info.pop("params"))
        model.load_state_dict(model_info.pop("model_state_dict"))
        return model


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
        loss = self.evaluate_model(evaluation_loader)
        self.update_evaluation_loss(loss)

        epoch_idx, checkpoint_idx = 0, 0
        checkpoint_idx = self.save_intermediate_model(epoch_idx, checkpoint_idx)

        logger.info("Training...")
        for epoch_idx in range(1, self._number_of_epochs + 1):
            logger.info(f"Epoch: {epoch_idx} / {self._number_of_epochs}")

            loss = self.train_model(training_loader)
            logger.info(f"Training loss (running average): {loss}")

            logger.info("Evaluating intermediate model...")
            loss = self.evaluate_model(evaluation_loader)
            self.update_evaluation_loss(loss)

            checkpoint_idx = self.save_intermediate_model(
                epoch_idx, checkpoint_idx
            )

        logger.info("Training is completed")
        self.save_model(self._number_of_epochs)

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
                self.save_model(epoch_idx, checkpoint_idx)
                return checkpoint_idx + 1
        return checkpoint_idx

    def save_model(
        self, epoch_idx: int, checkpoint_idx: Optional[int] = None
    ) -> None:
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
        # learning_process_info = self.create_learning_process_info(epoch_idx)
        checkpoint_info = self.create_checkpoint_info()
        checkpoint_info.save(checkpoint_filepath)
        logger.debug(f"checkpoint info saved to {checkpoint_filepath}")

    # def create_learning_process_info(
    #     self, epoch_idx: int
    # ) -> LearningProcessInfo:
    #     self._epoch_history["epoch"].append(epoch_idx)
    #     self._epoch_history["iteration"].append(self._iteration_idx)
    #     learning_process_info = LearningProcessInfo(
    #         epoch=epoch_idx,
    #         number_of_epochs=self._number_of_epochs,
    #     )
    #     return learning_process_info

    def create_checkpoint_info(self) -> CheckpointInfo:
        checkpoint_info = CheckpointInfo(
            epoch_history=self._epoch_history,
            evaluation_loss_history=self._evaluation_loss_history,
            training_loss_history=self._training_loss_history,
        )
        return checkpoint_info
