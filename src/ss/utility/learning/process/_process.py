from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    Self,
    Set,
    Tuple,
)

from collections import defaultdict
from enum import StrEnum, auto
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from ss.utility.assertion.validator import ReservedKeyNameValidator
from ss.utility.device import DeviceManager
from ss.utility.learning import module as Module
from ss.utility.learning import serialization
from ss.utility.learning.process.checkpoint import Checkpoint, CheckpointInfo
from ss.utility.learning.process.config import (
    EvaluationConfigProtocol,
    TestingConfig,
    TrainingConfig,
)
from ss.utility.learning.process.inference import InferenceContext
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class BaseLearningProcess:

    class Mode(StrEnum):
        TRAINING = auto()
        ANALYSIS = auto()
        INFERENCE = auto()

    def __init__(
        self,
        module: Module.BaseLearningModule,
        loss_function: Callable[..., torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self._device_manager = DeviceManager()
        self._module = self._device_manager.load_module(module)
        self._loss_function = loss_function
        self._optimizer = optimizer

        self._iteration: int = 0
        self._epoch: int = 0
        self._training_loss: float = 0.0

        self._epoch_history: DefaultDict[str, List[int]] = defaultdict(list)
        self._training_loss_history: DefaultDict[str, List[float]] = (
            defaultdict(list)
        )
        self._validation_loss_history: DefaultDict[str, List[float]] = (
            defaultdict(list)
        )

    def _update_epoch(self, max_epoch: int) -> None:
        self._epoch_history["iteration"].append(self._iteration)
        self._epoch_history["epoch"].append(self._epoch)
        logger.info(f"Finish epoch: {self._epoch} / {max_epoch}")
        logger.info("")

    def _update_validation_loss(self, losses: NDArray) -> None:
        self._validation_loss_history["iteration"].append(self._iteration)
        loss_mean, loss_std = float(losses.mean()), float(losses.std())
        self._validation_loss_history["loss_mean"].append(loss_mean)
        self._validation_loss_history["loss_std"].append(loss_std)
        logger.info(f"Validation loss: {loss_mean}" " \xb1 " f"{loss_std}")
        # \xb1 is a unicode character for the plus-minus sign (±)

    def _update_training_loss(self, loss: float) -> None:
        self._training_loss_history["iteration"].append(self._iteration)
        self._training_loss_history["loss"].append(loss)

    def _evaluate_one_batch(
        self, data_batch: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """
        Evaluate one batch of data and return the loss tensor.

        Parameters
        ----------
        data_batch : Tuple[torch.Tensor, ...]
            One batch of data.

        Returns
        -------
        loss_tensor : torch.Tensor
            The loss tensor evaluated from the given data batch.

        Raises
        ------
        NotImplementedError
            This method is required to be implemented in the derived class.
        """
        raise NotImplementedError

    def evaluate_model(
        self,
        data_loader: DataLoader[Tuple[torch.Tensor, ...]],
        evaluation_config: EvaluationConfigProtocol,
    ) -> NDArray:

        with self._module.evaluation_mode():
            logger.info("Evaluating model...")
            losses = []
            # with torch.no_grad():
            for b, data_batch in logger.progress_bar(
                enumerate(data_loader), total=len(data_loader)
            ):
                if evaluation_config.termination_condition(
                    batch_number=b
                ).satisfied():
                    break

                evaluation_loss = self._evaluate_one_batch(
                    self._device_manager.load_data_batch(data_batch)
                ).item()

                losses.append(evaluation_loss)

        return np.array(losses)

    def _train_one_batch(
        self, data_batch: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        self._optimizer.zero_grad()
        loss_tensor = self._evaluate_one_batch(
            self._device_manager.load_data_batch(data_batch)
        )
        loss_tensor.backward()
        self._optimizer.step()
        return loss_tensor

    def train_model(
        self,
        training_data_loader: DataLoader[Tuple[torch.Tensor, ...]],
        validation_data_loader: DataLoader[Tuple[torch.Tensor, ...]],
        training_config: TrainingConfig,
    ) -> None:
        with self._module.training_mode():
            logger.info("Training one epoch...")
            for data_batch in logger.progress_bar(
                training_data_loader, total=len(training_data_loader)
            ):
                training_loss = self._train_one_batch(data_batch).item()
                self._update_training_loss(training_loss)

                self._iteration += 1
                self._training_loss = (training_loss + self._training_loss) / 2

                if training_config.validation.condition(
                    iteration=self._iteration,
                ).satisfied():
                    validation_losses = self.evaluate_model(
                        validation_data_loader, training_config.validation
                    )
                    self._update_validation_loss(validation_losses)

                if training_config.termination.condition(
                    iteration=self._iteration
                ).satisfied():
                    break
            else:
                self._epoch += 1
        logger.info(f"Training loss (running average): {self._training_loss}")

    def training(
        self,
        training_data_loader: DataLoader[Tuple[torch.Tensor, ...]],
        validation_data_loader: DataLoader[Tuple[torch.Tensor, ...]],
        training_config: TrainingConfig,
    ) -> None:
        self._checkpoint = Checkpoint(training_config.checkpoint)

        try:

            if training_config.validation.at_initial:
                logger.info("Model evaluation before training...")
                validation_losses = self.evaluate_model(
                    validation_data_loader, training_config.validation
                )
                self._update_validation_loss(validation_losses)

                self._update_epoch(training_config.termination.max_epoch)
                self._checkpoint.save(
                    self._module,
                    self.save_checkpoint_info(),
                    self.save_model_info(),
                )

            logger.info("Start training...")
            while training_config.termination.condition(
                epoch=self._epoch
            ).not_satisfied():

                self.train_model(
                    training_data_loader,
                    validation_data_loader,
                    training_config,
                )

                self._update_epoch(training_config.termination.max_epoch)

                if training_config.checkpoint.condition(
                    epoch=self._epoch
                ).satisfied():
                    self._checkpoint.save(
                        self._module,
                        self.save_checkpoint_info(),
                        self.save_model_info(),
                    )

        except KeyboardInterrupt:
            training_config.termination.reason = (
                training_config.termination.TerminationReason.USER_INTERRUPT
            )

        finally:
            logger.info(
                f"Training is terminated with the reason: {training_config.termination.reason} triggered!"
            )

        self._checkpoint.finalize().save(
            self._module,
            self.save_checkpoint_info(),
            self.save_model_info(),
        )

    def test_model(
        self,
        testing_data_loader: DataLoader[Tuple[torch.Tensor, ...]],
        testing_config: Optional[TestingConfig] = None,
    ) -> None:
        if testing_config is None:
            testing_config = TestingConfig()
        logger.info("Testing...")
        testing_losses = self.evaluate_model(
            testing_data_loader, testing_config
        )
        loss_mean, loss_std = float(testing_losses.mean()), float(
            testing_losses.std()
        )
        logger.info(
            f"Testing is completed with loss: {loss_mean}"
            " \xb1 "
            f"{loss_std}"
        )

    def _save_checkpoint_info(self) -> CheckpointInfo:
        """
        Save custom checkpoint information. This method can be overridden in the derived class.

        Returns
        -------
        custom_checkpoint_info: CheckpointInfo
            Custom checkpoint information.
        """
        custom_checkpoint_info = CheckpointInfo()
        return custom_checkpoint_info

    def save_checkpoint_info(self) -> CheckpointInfo:
        checkpoint_info = CheckpointInfo(
            __iteration__=self._iteration,
            __epoch__=self._epoch,
            __training_loss__=self._training_loss,
            __epoch_history__=self._epoch_history,
            __validation_loss_history__=self._validation_loss_history,
            __training_loss_history__=self._training_loss_history,
        )
        custom_checkpoint_info = self._save_checkpoint_info()
        ReservedKeyNameValidator(
            custom_checkpoint_info, checkpoint_info.keys()
        )
        checkpoint_info.update(custom_checkpoint_info)
        return checkpoint_info

    def _load_checkpoint_info(self, checkpoint_info: CheckpointInfo) -> None:
        """
        Load custom checkpoint information. This method can be overridden in the derived class.

        Parameters
        ----------
        checkpoint_info : CheckpointInfo
            Custom checkpoint information to be loaded.
        """
        pass

    def load_checkpoint_info(
        self,
        checkpoint_info: CheckpointInfo,
    ) -> None:

        self._iteration = checkpoint_info.pop("__iteration__")
        self._epoch = checkpoint_info.pop("__epoch__")
        self._training_loss = checkpoint_info.pop("__training_loss__")

        self._epoch_history = checkpoint_info.pop("__epoch_history__")
        self._validation_loss_history = checkpoint_info.pop(
            "__validation_loss_history__"
        )
        self._training_loss_history = checkpoint_info.pop(
            "__training_loss_history__"
        )

        self._load_checkpoint_info(checkpoint_info)

    def _save_model_info(self) -> Dict[str, Any]:
        """
        Save custom model information. This method can be overridden in the derived class.

        Returns
        -------
        custom_model_info: Dict[str, Any]
            Custom model information.
        """
        custom_model_info: Dict[str, Any] = dict()
        return custom_model_info

    def save_model_info(self) -> Dict[str, Any]:
        model_info = dict(
            __loss_function__=self._loss_function,
            __optimizer__=self._optimizer,
        )
        custom_model_info = self._save_model_info()
        ReservedKeyNameValidator(custom_model_info, model_info.keys())
        model_info.update(custom_model_info)
        return model_info

    @classmethod
    def load_model_info(
        cls, model_info: Dict[str, Any]
    ) -> Tuple[
        Callable[..., torch.Tensor], torch.optim.Optimizer, Dict[str, Any]
    ]:
        loss_function = model_info.pop("__loss_function__")
        optimizer = model_info.pop("__optimizer__")
        return loss_function, optimizer, model_info

    @classmethod
    def from_checkpoint(
        cls,
        module: Module.BaseLearningModule,
        model_filepath: Path,
        safe_callables: Optional[Set[serialization.SafeCallable]] = None,
    ) -> Self:
        module, model_info, checkpoint_info = Checkpoint.load(
            module, model_filepath, safe_callables
        )
        loss_function, optimizer, model_info = cls.load_model_info(model_info)
        optimizer.param_groups = []
        optimizer.add_param_group({"params": module.parameters()})
        learning_process = cls(
            module=module,
            loss_function=loss_function,
            optimizer=optimizer,
            **model_info,
        )
        learning_process.load_checkpoint_info(checkpoint_info)
        logger.info(f"Learning process is loaded from {model_filepath}")
        logger.info(
            f"with epoch: {learning_process._epoch} and iteration: {learning_process._iteration}"
        )
        return learning_process

    @classmethod
    def inference_mode(
        cls, *modules: Module.BaseLearningModule
    ) -> InferenceContext:
        return InferenceContext(*modules)
