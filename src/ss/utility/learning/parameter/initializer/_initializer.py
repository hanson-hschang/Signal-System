from dataclasses import dataclass
from typing import Protocol, TypeVar

import torch

from ss.utility.learning.module.config import BaseLearningConfig

I = TypeVar("I", bound="Initializer")  # noqa: E741


class InitializerProtocol(Protocol):
    def __call__(self, shape: tuple[int, ...]) -> torch.Tensor: ...


@dataclass
class Initializer(BaseLearningConfig):
    def __call__(self, shape: tuple[int, ...]) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def basic_config(cls: type[I]) -> I:
        raise NotImplementedError
