from typing import DefaultDict, List

from collections import defaultdict

from numpy.typing import NDArray

from ss.utility.assertion.validator import ReservedKeyNameValidator
from ss.utility.learning.process.checkpoint import CheckpointInfo
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class LearningProcessInfoMixin:

    def __init__(self) -> None:
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

    @property
    def validation_count(self) -> int:
        """
        Get the number of validation iterations.

        Returns
        -------
        validation_count : int
            The number of validation iterations.
        """
        return max(len(self._validation_loss_history["iteration"]) - 1, 0)

    def _record_epoch(self, max_epoch: int) -> None:
        self._epoch_history["iteration"].append(self._iteration)
        self._epoch_history["epoch"].append(self._epoch)
        logger.info(f"Finish epoch: {self._epoch} / {max_epoch}")
        logger.info("")

    def _record_validation_loss(self, losses: NDArray) -> None:
        self._validation_loss_history["iteration"].append(self._iteration)
        loss_mean, loss_std = float(losses.mean()), float(losses.std())
        self._validation_loss_history["loss_mean"].append(loss_mean)
        self._validation_loss_history["loss_std"].append(loss_std)
        logger.info(f"Validation loss: {loss_mean}" " \xb1 " f"{loss_std}")
        # \xb1 is a unicode character for the plus-minus sign (±)

    def _record_training_loss(self, loss: float) -> None:
        self._training_loss_history["iteration"].append(self._iteration)
        self._training_loss_history["loss"].append(loss)

    def record(self) -> None:
        """
        Record training information.
        """
        pass

    def save_checkpoint_info(self) -> CheckpointInfo:
        """
        Save custom checkpoint information. This method can be overridden in the derived class.

        Returns
        -------
        custom_checkpoint_info: CheckpointInfo
            Custom checkpoint information.
        """
        custom_checkpoint_info = CheckpointInfo()
        return custom_checkpoint_info

    def _save_checkpoint_info(self) -> CheckpointInfo:
        checkpoint_info = CheckpointInfo(
            __iteration__=self._iteration,
            __epoch__=self._epoch,
            __training_loss__=self._training_loss,
            __epoch_history__=self._epoch_history,
            __validation_loss_history__=self._validation_loss_history,
            __training_loss_history__=self._training_loss_history,
        )
        custom_checkpoint_info = self.save_checkpoint_info()
        ReservedKeyNameValidator(
            custom_checkpoint_info, checkpoint_info.keys()
        )
        checkpoint_info.update(custom_checkpoint_info)
        return checkpoint_info

    def load_checkpoint_info(self, checkpoint_info: CheckpointInfo) -> None:
        """
        Load custom checkpoint information. This method can be overridden in the derived class.

        Parameters
        ----------
        checkpoint_info : CheckpointInfo
            Custom checkpoint information to be loaded.
        """
        pass

    def _load_checkpoint_info(
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

        self.load_checkpoint_info(checkpoint_info)
