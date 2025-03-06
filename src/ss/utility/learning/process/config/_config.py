from typing import Optional, Protocol, Self

from dataclasses import dataclass, field
from enum import Enum, Flag, auto

from ss.utility.condition import Condition
from ss.utility.learning.process.checkpoint.config import CheckpointConfig
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class EvaluationConfigProtocol(Protocol):
    def termination_condition(
        self, batch_number: Optional[int] = None
    ) -> Condition: ...


@dataclass
class TestingConfig:
    max_batch: Optional[int] = None

    def __post_init__(self) -> None:
        self._termination_condition = Condition(any)

    def termination_condition(
        self, batch_number: Optional[int] = None
    ) -> Condition:
        if batch_number is not None:
            self._termination_condition(
                max_batch=(
                    self.max_batch is not None
                    and batch_number >= self.max_batch
                )
            )
        return self._termination_condition


@dataclass
class ValidationConfig:
    per_iteration_period: int = 1
    at_initial: bool = True
    max_batch: Optional[int] = None

    def __post_init__(self) -> None:
        self._condition = Condition(any)
        self._termination_condition = Condition(any)

    def condition(self, iteration: Optional[int] = None) -> Condition:
        if iteration is not None:
            self._condition(
                iteration=(iteration % self.per_iteration_period) == 0
            )
        return self._condition

    def termination_condition(
        self, batch_number: Optional[int] = None
    ) -> Condition:
        if batch_number is not None:
            self._termination_condition(
                max_batch=(
                    self.max_batch is not None
                    and batch_number >= self.max_batch
                )
            )
        return self._termination_condition


@dataclass
class TerminationConfig:

    class TerminationReason(Flag):
        NOT_TERMINATED = 0
        MAX_EPOCH = auto()
        MAX_ITERATION = auto()
        USER_INTERRUPT = auto()
        # MAX_NO_IMPROVEMENT = auto()

    max_epoch: int = 1
    max_iteration: Optional[int] = None
    # max_no_improvement: Optional[int] = None

    def __post_init__(self) -> None:
        self._condition = Condition(any)
        self._termination_reason = self.TerminationReason.NOT_TERMINATED
        self._update_condition()

    def _update_condition(
        self,
        max_epoch: bool = False,
        max_iteration: bool = False,
        user_interrupt: bool = False,
    ) -> None:
        if max_epoch:
            self._termination_reason = (
                self.TerminationReason.MAX_EPOCH
                if self._termination_reason
                == self.TerminationReason.NOT_TERMINATED
                else self._termination_reason
                | self.TerminationReason.MAX_EPOCH
            )
        if max_iteration:
            self._termination_reason = (
                self.TerminationReason.MAX_ITERATION
                if self._termination_reason
                == self.TerminationReason.NOT_TERMINATED
                else self._termination_reason
                | self.TerminationReason.MAX_ITERATION
            )
        if user_interrupt:
            self._termination_reason = self.TerminationReason.USER_INTERRUPT
        self._condition(
            max_epoch=max_epoch,
            max_iteration=max_iteration,
            user_interrupt=user_interrupt,
        )

    @property
    def reason(self) -> TerminationReason:
        return self._termination_reason

    @reason.setter
    def reason(self, reason: TerminationReason) -> None:
        if reason == self.TerminationReason.USER_INTERRUPT:
            self._update_condition(user_interrupt=True)
            return
        logger.error(
            f"Cannot manually set the termination reason unless it is a user interruption."
        )

    def condition(
        self,
        epoch: Optional[int] = None,
        iteration: Optional[int] = None,
        # loss: Optional[float] = None,
    ) -> Condition:
        # Once the termination is flagged, it will not be update
        if self._termination_reason != self.TerminationReason.NOT_TERMINATED:
            return self._condition

        # Update the condition status
        self._update_condition(
            max_epoch=(epoch is not None and epoch >= self.max_epoch),
            max_iteration=(
                self.max_iteration is not None
                and iteration is not None
                and iteration >= self.max_iteration
            ),
        )
        return self._condition

    def reset(self) -> None:
        self._termination_reason = self.TerminationReason.NOT_TERMINATED
        self._update_condition()


@dataclass
class TrainingConfig:
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
