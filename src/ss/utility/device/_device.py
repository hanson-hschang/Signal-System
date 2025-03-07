from typing import TypeVar

from enum import StrEnum, auto

import torch

from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


M = TypeVar("M", bound=torch.nn.Module)


class Device(StrEnum):
    CUDA = auto()
    CPU = auto()
    MPS = auto()

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self)
