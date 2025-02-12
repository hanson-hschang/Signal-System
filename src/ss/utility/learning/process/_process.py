from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    Self,
    Tuple,
    Union,
)

from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from ss.utility.assertion.validator import (
    FilePathValidator,
    NonnegativeIntegerValidator,
    PositiveIntegerValidator,
)
from ss.utility.device import DeviceManager
from ss.utility.learning.module import BaseLearningModule
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


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

    def __init__(
        self,
        model: BaseLearningModule,
        loss_function: Callable[..., torch.Tensor],
        optimizer: torch.optim.Optimizer,
        number_of_epochs: int,
        model_filename: Union[str, Path],
        evaluate_model_iteration_skip: int = 1,
        save_model_epoch_skip: int = 0,
    ) -> None:
        self._device_manager = DeviceManager()
        self._model = self._device_manager.load_module(model)
        self._loss_function = loss_function
        self._optimizer = optimizer
        self._number_of_epochs = PositiveIntegerValidator(
            number_of_epochs
        ).get_value()
        self._model_filepath = FilePathValidator(
            model_filename,
            BaseLearningModule.MODEL_FILE_EXTENSION,
        ).get_filepath()
        self._evaluate_model_iteration_skip = PositiveIntegerValidator(
            evaluate_model_iteration_skip
        ).get_value()
        self._save_model_epoch_skip = NonnegativeIntegerValidator(
            save_model_epoch_skip
        ).get_value()

        self._iteration_idx: int = 0
        self._epoch_history: DefaultDict[str, List[int]] = defaultdict(list)
        self._training_loss = 0.0
        self._evaluation_loss_history: DefaultDict[str, List[float]] = (
            defaultdict(list)
        )
        self._training_loss_history: DefaultDict[str, List[float]] = (
            defaultdict(list)
        )

        self._save_intermediate_models = self._save_model_epoch_skip > 0
        if self._save_intermediate_models:
            self._digits_of_number_of_checkpoints = (
                len(str(self._number_of_epochs // self._save_model_epoch_skip))
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

    def _evaluate_one_batch(
        self, data_batch: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        raise NotImplementedError

    def evaluate_model(
        self, data_loader: DataLoader[Tuple[torch.Tensor, ...]]
    ) -> float:
        with self._model.evaluation_mode():
            logger.info("Evaluating model...")
            loss = 0.0
            with torch.no_grad():
                for i, data_batch in logger.progress_bar(
                    enumerate(data_loader), total=len(data_loader)
                ):
                    loss += self._evaluate_one_batch(
                        self._device_manager.load_data_batch(data_batch)
                    ).item()
            loss /= i + 1
        return loss

    def _train_one_batch(
        self, data_batch: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        self._optimizer.zero_grad()
        _loss = self._evaluate_one_batch(
            self._device_manager.load_data_batch(data_batch)
        )
        _loss.backward()
        self._optimizer.step()
        return _loss

    def train_model(
        self,
        data_loader: DataLoader[Tuple[torch.Tensor, ...]],
        evaluation_loader: DataLoader[Tuple[torch.Tensor, ...]],
    ) -> float:
        with self._model.training_mode():
            logger.info("Training one epoch...")
            for data_batch in logger.progress_bar(
                data_loader, total=len(data_loader)
            ):
                training_loss = self._train_one_batch(data_batch).item()
                self.update_training_loss(training_loss)
                self._training_loss = (training_loss + self._training_loss) / 2
                self._iteration_idx += 1

                if (
                    self._iteration_idx % self._evaluate_model_iteration_skip
                    == 0
                ):
                    evaluation_loss = self.evaluate_model(evaluation_loader)
                    self.update_evaluation_loss(evaluation_loss)
        return self._training_loss

    def train(
        self,
        training_loader: DataLoader[Tuple[torch.Tensor, ...]],
        evaluation_loader: DataLoader[Tuple[torch.Tensor, ...]],
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

            loss = self.train_model(training_loader, evaluation_loader)
            logger.info(f"Training loss (running average): {loss}")

            self.update_epoch(epoch_idx)
            checkpoint_idx = self.save_intermediate_model(
                epoch_idx, checkpoint_idx
            )

        logger.info("Training is completed")
        self.save_model()

    def test_model(
        self,
        testing_loader: DataLoader[Tuple[torch.Tensor, ...]],
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
                    f"Intermediate models are saved every {self._save_model_epoch_skip} epoch(s)"
                )
            if epoch_idx % self._save_model_epoch_skip == 0:
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
            trained_epochs=self._number_of_epochs,
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

    def load_model(self, filename: Union[str, Path]) -> None:
        model, model_info = self._model.load(filename)
        self._model = self._device_manager.load_module(model)
        self._number_of_epochs = model_info["trained_epochs"]
