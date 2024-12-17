from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Self,
    Type,
    TypeVar,
    Union,
)

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

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
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class Mode(StrEnum):
    TRAIN = "TRAIN"
    VISUALIZATION = "VISUALIZATION"
    INFERENCE = "INFERENCE"


MODEL_FILE_EXTENSION = (".pt", ".pth")

BLP = TypeVar("BLP", bound="BaseLearningParameters")


@dataclass
class BaseLearningParameters:
    pass


BLM = TypeVar("BLM", bound="BaseLearningModule")


class BaseLearningModule(nn.Module):

    def __init__(self, params: BaseLearningParameters) -> None:
        super().__init__()
        assert issubclass(
            type(params), BaseLearningParameters
        ), f"{type(params) = } must be a subclass of {BaseLearningParameters}"
        self._params = params

    def save(self, filename: Union[str, Path], **checkpoint_info: Any) -> None:
        filepath = FilePathValidator(
            filename, MODEL_FILE_EXTENSION
        ).get_filepath()
        assert (
            "params" not in checkpoint_info
        ), "'params' is a reserved key for **checkpoint_info."
        assert (
            "model_state_dict" not in checkpoint_info
        ), "'model_state_dict' is a reserved key for **checkpoint_info."
        checkpoint_info["params"] = self._params
        checkpoint_info["model_state_dict"] = self.state_dict()
        torch.save(checkpoint_info, filepath)

    @classmethod
    def load(cls: Type[BLM], filename: Union[str, Path]) -> BLM:
        filepath = FilePathValidator(
            filename, MODEL_FILE_EXTENSION
        ).get_filepath()
        checkpoint_info: Dict[str, Any] = torch.load(
            filepath, weights_only=True
        )
        model = cls(checkpoint_info.pop("params"))
        model.load_state_dict(checkpoint_info.pop("model_state_dict"))
        return model


class CheckpointInfo(dict):

    @classmethod
    def load(cls, filename: Union[str, Path]) -> Self:
        filepath = FilePathValidator(
            filename, MODEL_FILE_EXTENSION
        ).get_filepath()
        checkpoint_info = torch.load(filepath, weights_only=True)
        return cls(**checkpoint_info)


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
            MODEL_FILE_EXTENSION,
        ).get_filepath()
        self._save_model_step_skip = self._SaveModelEpochSkipValidator(
            save_model_epoch_skip
        ).get_value()

        self._evaluation_loss = 0.0
        self._training_loss = 0.0
        self._training_loss_history: List[float] = []

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

    def _train_one_batch(self, data_batch: Any) -> float:
        raise NotImplementedError

    def train_one_epoch(self, data_loader: DataLoader) -> None:
        self._model.train()
        with logging_redirect_tqdm(loggers=[logger]):
            for i, data_batch in tqdm(
                enumerate(data_loader), total=len(data_loader)
            ):
                loss = self._train_one_batch(data_batch)
                self._training_loss_history.append(loss)
                self._training_loss = (loss + self._training_loss) / 2
        logger.info(f"Training loss (running average): {self._training_loss}")

    def _evaluate_one_batch(self, data_batch: Any) -> float:
        raise NotImplementedError

    def evaluate_model(self, data_loader: DataLoader) -> float:
        self._model.eval()
        _loss = 0.0
        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                loss = self._evaluate_one_batch(data_batch)
                _loss += loss
        _loss /= i + 1
        return _loss

    def train(
        self,
        training_loader: DataLoader,
        evaluation_loader: DataLoader,
    ) -> None:
        epoch_idx, checkpoint_idx = 0, 0
        self.evaluate_model(evaluation_loader)
        logger.info("Training...")
        checkpoint_idx = self.save_intermediate_model(epoch_idx, checkpoint_idx)
        for epoch_idx in range(1, self._number_of_epochs + 1):
            logger.info(f"Epoch: {epoch_idx} / {self._number_of_epochs}")
            self.train_one_epoch(training_loader)
            self._evaluation_loss = self.evaluate_model(evaluation_loader)
            logger.info(f"Evaluation loss: {self._evaluation_loss}")
            checkpoint_idx = self.save_intermediate_model(
                epoch_idx, checkpoint_idx
            )
        logger.info("Training is completed")
        self.save_model(self._number_of_epochs)

    def test(
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
                    f"Intermediate models are saved every {self._save_model_step_skip} epochs"
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

        checkpoint_info = self.create_checkpoint_info(epoch_idx)
        self._model.save(model_filepath, **checkpoint_info)
        logger.debug(f"model saved to {model_filepath}")

    def create_checkpoint_info(self, epoch_idx: int) -> CheckpointInfo:
        checkpoint_info = CheckpointInfo(
            epoch=epoch_idx,
            number_of_epochs=self._number_of_epochs,
            evaluation_loss=self._evaluation_loss,
            training_loss=self._training_loss,
            training_loss_history=self._training_loss_history,
        )
        return checkpoint_info
