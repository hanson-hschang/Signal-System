from typing import Protocol

import torch
from torch.optim import lr_scheduler
from numpy.typing import NDArray


class StepSchedulerProtocol(Protocol):
    def condition(
        self,
        iteration: int | None = None,
        losses: NDArray | None = None,
        epoch: int | None = None,
    ) -> None: ...

    def step(self) -> None: ...


class BaseStepScheduler:
    def __init__(self) -> None:
        self._step_on_iteration: bool = False
        self._step_on_epoch: bool = False
        self._step_on_validation: bool = False
        self.reset()

    def reset(self) -> None:
        self._iteration: int | None = None
        self._validation_losses: NDArray | None = None
        self._epoch: int | None = None

    def condition(
        self,
        iteration: int | None = None,
        losses: NDArray | None = None,
        epoch: int | None = None,
    ) -> None:
        self._iteration = iteration
        self._validation_losses = losses
        self._epoch = epoch

    def _step_iteration(self) -> None: ...

    def _step_validation(self, validation_losses: NDArray) -> None: ...

    def _step_epoch(self) -> None: ...

    def step(self) -> None:
        if self._step_on_iteration and self._iteration is not None:
            self._step_iteration()
        if self._step_on_validation and self._validation_losses is not None:
            self._step_validation(self._validation_losses)
        if self._step_on_epoch and self._epoch is not None:
            self._step_epoch()
        self.reset()

    @staticmethod
    def from_torch(
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        step_on_iteration: bool = False,
    ) -> "TorchStepScheduler":
        """
        Create a TorchStepScheduler from a PyTorch learning rate scheduler.
        """
        return TorchStepScheduler(
            scheduler, step_on_iteration=step_on_iteration
        )


class TorchStepScheduler(BaseStepScheduler):
    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        step_on_iteration: bool = False,
    ) -> None:
        self._torch_lr_scheduler = scheduler
        super().__init__()
        if step_on_iteration:
            self._step_on_iteration = True
        else:
            self._step_on_epoch = True
        if isinstance(
            scheduler,
            lr_scheduler.ReduceLROnPlateau,
        ):
            self._step_on_validation = True
            self._step_on_epoch = False
            self._step_on_iteration = False

    def _step_iteration(self) -> None:
        self._torch_lr_scheduler.step()

    def _step_epoch(self) -> None:
        self._torch_lr_scheduler.step()

    def _step_validation(self, validation_losses: NDArray) -> None:
        self._torch_lr_scheduler.step(validation_losses.mean())


class StepScheduler(BaseStepScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self._optimizer = optimizer
        super().__init__()
