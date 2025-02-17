from typing import Optional, Self

from dataclasses import dataclass, field
from enum import Enum, Flag, auto

from ss.utility.condition import Condition
from ss.utility.learning.process.checkpoint.config import CheckpointConfig


@dataclass
class EvaluationConfig:
    per_iteration_period: int = 1

    def __post_init__(self) -> None:
        self._condition = Condition(any)

    def condition(self, iteration: Optional[int] = None) -> Condition:
        if iteration is not None:
            self._condition(
                iteration=(iteration % self.per_iteration_period) == 0
            )
        return self._condition


@dataclass
class TerminationConfig:

    class TerminationReason(Flag):
        NOT_TERMINATED = 0
        MAX_EPOCH = auto()
        MAX_ITERATION = auto()
        # MAX_NO_IMPROVEMENT = auto()

    max_epoch: int = 1
    max_iteration: Optional[int] = None
    # max_no_improvement: Optional[int] = None

    def __post_init__(self) -> None:
        self._condition = Condition(any)
        self._termination_reason = self.TerminationReason.NOT_TERMINATED
        self._update_condition()

    @property
    def reason(self) -> TerminationReason:
        return self._termination_reason

    def _update_condition(
        self,
        max_epoch: bool = False,
        max_iteration: bool = False,
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
        self._condition(
            max_epoch=max_epoch,
            max_iteration=max_iteration,
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
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
