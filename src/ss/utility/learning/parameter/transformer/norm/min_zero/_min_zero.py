import torch

from ss.utility.learning.parameter.transformer import Transformer
from ss.utility.learning.parameter.transformer.norm.min_zero import (
    config as Config,
)


class MinZeroNormTransformer(
    Transformer[Config.MinZeroNormTransformerConfig],
):
    def forward(self, parameter: torch.Tensor) -> torch.Tensor:
        min_values, _ = torch.min(parameter, dim=-1, keepdim=True)
        nonpositive_min_values = torch.min(min_values, torch.tensor(0.0))
        expanded_min = nonpositive_min_values.expand_as(parameter)
        offset_parameter = parameter - expanded_min
        norm = torch.norm(
            offset_parameter, p=self._config.order, dim=-1, keepdim=True
        )
        normed_parameter: torch.Tensor = offset_parameter / norm
        return normed_parameter

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        if value.numel() == 1:
            raise ValueError("value must have more than one element.")
        negative_mask = value < 0
        if negative_mask.any():
            raise ValueError("value must be non-negative.")
        norm = torch.norm(value, p=self._config.order, dim=-1)
        first_element = norm.reshape(-1)[0].item()
        reference = torch.full_like(norm, first_element)
        if not (torch.allclose(norm, reference) and first_element == 1):
            raise ValueError(
                f"value must be normalized with order = {self._config.order}."
            )
        return value
