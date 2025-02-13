from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, Tuple

from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ss.utility.device import DeviceManager
from ss.utility.learning import module as Module
from ss.utility.learning import serialization
from ss.utility.learning.process import config as Config
from ss.utility.learning.process.checkpoint import Checkpoint, CheckpointInfo
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class BaseLearningProcess:

    def __init__(
        self,
        model: Module.BaseLearningModule,
        loss_function: Callable[..., torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self._device_manager = DeviceManager()
        self._model = self._device_manager.load_module(model)
        self._loss_function = loss_function
        self._optimizer = optimizer

        self._iteration: int = 0
        self._epoch: int = 0
        self._training_loss: float = 0.0

        self._epoch_history: DefaultDict[str, List[int]] = defaultdict(list)
        self._training_loss_history: DefaultDict[str, List[float]] = (
            defaultdict(list)
        )
        self._evaluation_loss_history: DefaultDict[str, List[float]] = (
            defaultdict(list)
        )

    def _update_epoch(self, max_epoch: int) -> None:
        self._epoch_history["iteration"].append(self._iteration)
        self._epoch_history["epoch"].append(self._epoch)
        logger.info(f"Finish epoch: {self._epoch} / {max_epoch}")
        logger.info("")

    def _update_evaluation_loss(self, loss: float) -> None:
        self._evaluation_loss_history["iteration"].append(self._iteration)
        self._evaluation_loss_history["loss"].append(loss)
        logger.info(f"Evaluation loss: {loss}")

    def _update_training_loss(self, loss: float) -> None:
        self._training_loss_history["iteration"].append(self._iteration)
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
        training_config: Config.TrainingConfig,
    ) -> float:
        with self._model.training_mode():
            logger.info("Training one epoch...")
            for data_batch in logger.progress_bar(
                data_loader, total=len(data_loader)
            ):
                training_loss = self._train_one_batch(data_batch).item()
                self._update_training_loss(training_loss)
                self._training_loss = (training_loss + self._training_loss) / 2
                self._iteration += 1

                if training_config.evaluation.condition(
                    iteration=self._iteration,
                ).satisfied():
                    evaluation_loss = self.evaluate_model(evaluation_loader)
                    self._update_evaluation_loss(evaluation_loss)

                if training_config.termination.condition(
                    iteration=self._iteration
                ).satisfied():
                    break
            else:
                self._epoch += 1

        return self._training_loss

    def train(
        self,
        training_loader: DataLoader[Tuple[torch.Tensor, ...]],
        evaluation_loader: DataLoader[Tuple[torch.Tensor, ...]],
        training_config: Config.TrainingConfig,
    ) -> None:
        self._checkpoint = Checkpoint(training_config.checkpoint)

        logger.info("Model evaluation before training...")
        loss = self.evaluate_model(evaluation_loader)
        self._update_evaluation_loss(loss)

        self._update_epoch(training_config.termination.max_epoch)
        self._checkpoint.save(
            self._model,
            self._create_checkpoint_info(),
        )

        logger.info("Start training...")
        while not training_config.termination.condition(
            epoch=self._epoch
        ).satisfied():

            loss = self.train_model(
                training_loader,
                evaluation_loader,
                training_config,
            )
            logger.info(f"Training loss (running average): {loss}")

            self._update_epoch(training_config.termination.max_epoch)

            if training_config.checkpoint.condition(
                epoch=self._epoch
            ).satisfied():
                self._checkpoint.save(
                    self._model,
                    self._create_checkpoint_info(),
                )

        logger.info(
            f"Training is completed with {training_config.termination.reason} reached"
        )
        self._checkpoint.finalize().save(
            self._model,
            self._create_checkpoint_info(),
        )

    def test_model(
        self,
        testing_loader: DataLoader[Tuple[torch.Tensor, ...]],
    ) -> None:
        logger.info("Testing...")
        loss = self.evaluate_model(testing_loader)
        logger.info(f"Testing is completed with loss: {loss}")

    def _create_checkpoint_info(self) -> CheckpointInfo:
        checkpoint_info = CheckpointInfo(
            iteration=self._iteration,
            epoch=self._epoch,
            training_loss=self._training_loss,
            epoch_history=self._epoch_history,
            evaluation_loss_history=self._evaluation_loss_history,
            training_loss_history=self._training_loss_history,
        )
        return checkpoint_info

    def load_checkpoint(
        self,
        filepath: Path,
        safe_callables: Optional[Set[serialization.SafeCallable]] = None,
    ) -> Dict[str, Any]:
        model, module_info = self._model.load(
            filepath.with_suffix(Module.BaseLearningModule.FILE_EXTENSIONS[0]),
            safe_callables,
        )
        self._model = self._device_manager.load_module(model)
        checkpoint_info = CheckpointInfo.load(
            filepath.with_suffix(CheckpointInfo.FILE_EXTENSION)
        )
        self._iteration = checkpoint_info.get("iteration", 0)
        self._epoch = checkpoint_info.get("epoch", 0)
        self._training_loss = checkpoint_info.get("training_loss", 0.0)
        return module_info
