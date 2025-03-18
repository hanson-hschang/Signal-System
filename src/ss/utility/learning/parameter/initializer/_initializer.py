from typing import Any, Protocol, Tuple, Type, TypeVar

from dataclasses import dataclass

import torch

from ss.utility.learning.module.config import BaseLearningConfig

I = TypeVar("I", bound="Initializer")


class InitializerProtocol(Protocol):
    def __call__(self, shape: Tuple[int, ...]) -> torch.Tensor: ...


@dataclass
class Initializer(BaseLearningConfig):

    def __call__(self, shape: Tuple[int, ...]) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def basic_config(cls: Type[I]) -> I:
        raise NotImplementedError
