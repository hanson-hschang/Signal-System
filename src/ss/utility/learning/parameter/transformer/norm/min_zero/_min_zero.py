from typing import Self, Tuple

import torch

from ss.utility.learning.parameter.transformer import Transformer
from ss.utility.learning.parameter.transformer.norm.min_zero import (
    config as Config,
)


class MinZeroNormTransformer(
    Transformer[Config.MinZeroNormTransformerConfig],
):
    def __init__(
        self,
        config: Config.MinZeroNormTransformerConfig,
        shape: Tuple[int, ...],
    ) -> None:
        super().__init__(config, shape)
        self._even_power = self._config.even_power
        self._epsilon = self._config.epsilon
        self._order = self._config.order

    def bind_with(self, transformer: Self) -> None: ...

    def forward(self, parameter: torch.Tensor) -> torch.Tensor:
        squared_parameter = parameter**self._even_power + self._epsilon
        norm = torch.norm(
            squared_parameter, p=self._order, dim=-1, keepdim=True
        )
        normed_parameter: torch.Tensor = squared_parameter / norm
        return normed_parameter

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        if value.numel() == 1:
            raise ValueError("value must have more than one element.")
        negative_mask = value < 0
        if negative_mask.any():
            raise ValueError("value must be non-negative.")
        norm = torch.norm(value, p=self._order, dim=-1)
        first_element = norm.reshape(-1)[0].item()
        reference = torch.full_like(norm, first_element)
        if not (torch.allclose(norm, reference) and first_element == 1):
            raise ValueError(
                f"value must be normalized with order = {self._order}."
            )
        return value
