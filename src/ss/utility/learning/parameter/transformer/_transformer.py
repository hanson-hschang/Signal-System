from typing import Self, TypeVar

import torch

from ss.utility.learning.module import BaseLearningModule
from ss.utility.learning.parameter.transformer.config import TC

# C = TypeVar("C", bound=Config.TransformerConfig)


class Transformer(BaseLearningModule[TC]):
    def __init__(self, config: TC, shape: tuple[int, ...]) -> None:
        super().__init__(config)
        self._shape = shape

    def bind_with(self, transformer: Self) -> None:
        raise NotImplementedError

    def forward(self, parameter: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


T = TypeVar("T", bound=Transformer)
