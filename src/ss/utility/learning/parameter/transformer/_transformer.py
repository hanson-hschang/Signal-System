from typing import Tuple, TypeVar

import torch

from ss.utility.learning.module import BaseLearningModule
from ss.utility.learning.parameter.transformer import config as Config

C = TypeVar("C", bound=Config.TransformerConfig)


class Transformer(BaseLearningModule[C]):
    def __init__(self, config: C, shape: Tuple[int, ...]) -> None:
        super().__init__(config)
        self._shape = shape

    def forward(self, parameter: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
