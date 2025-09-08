from typing import Self, TypeVar

import torch

from ss.utility.learning.parameter.transformer import Transformer
from ss.utility.learning.parameter.transformer.exp.config import (
    ExpTransformerConfig,
)


class ExpTransformer(
    Transformer[ExpTransformerConfig],
):
    def bind_with(self, transformer: Self) -> None: ...

    @torch.compile
    def forward(self, parameter: torch.Tensor) -> torch.Tensor:
        return torch.exp(parameter)

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        return torch.log(value)


ExpT = TypeVar("ExpT", bound=Transformer, default=ExpTransformer)
