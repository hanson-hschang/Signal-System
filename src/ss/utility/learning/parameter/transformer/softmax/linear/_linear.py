from typing import Generic, Self, Tuple, TypeVar, cast

import torch

from ss.utility.learning.parameter.positive import PositiveParameter
from ss.utility.learning.parameter.transformer import Transformer
from ss.utility.learning.parameter.transformer.exp import ExpT
from ss.utility.learning.parameter.transformer.exp.config import ExpTC
from ss.utility.learning.parameter.transformer.softmax.linear.config import (
    LinearSoftmaxTransformerConfig,
)


class LinearSoftmaxTransformer(
    Transformer[LinearSoftmaxTransformerConfig[ExpTC]],
    Generic[ExpT, ExpTC],
):
    def __init__(
        self,
        config: LinearSoftmaxTransformerConfig[ExpTC],
        shape: Tuple[int, ...],
    ) -> None:
        super().__init__(config, shape)
        self._intercept: PositiveParameter[ExpT, ExpTC] = (
            self._init_intercept()
        )
        self._slope: PositiveParameter[ExpT, ExpTC] = self._init_slope()

    def _init_intercept(self) -> PositiveParameter[ExpT, ExpTC]:
        shape = (1,) if len(self._shape) == 1 else self._shape[:-1] + (1,)
        return PositiveParameter[ExpT, ExpTC](self._config.intercept, shape)

    def _init_slope(self) -> PositiveParameter[ExpT, ExpTC]:
        shape = (1,) if len(self._shape) == 1 else self._shape[:-1] + (1,)
        return PositiveParameter[ExpT, ExpTC](self._config.slope, shape)

    @property
    def intercept_parameter(self) -> PositiveParameter[ExpT, ExpTC]:
        return self._intercept

    @property
    def intercept(self) -> torch.Tensor:
        intercept: torch.Tensor = self._intercept()[..., 0]
        return intercept

    @intercept.setter
    def intercept(self, value: torch.Tensor) -> None:
        if value.ndim == (len(self._shape) - 1):
            value = value.unsqueeze(-1)
        self._intercept.set_value(value)

    @property
    def slope_parameter(self) -> PositiveParameter[ExpT, ExpTC]:
        return self._slope

    @property
    def slope(self) -> torch.Tensor:
        slope: torch.Tensor = self._slope()[..., 0]
        return slope

    @slope.setter
    def slope(self, value: torch.Tensor) -> None:
        if value.ndim == (len(self._shape) - 1):
            value = value.unsqueeze(-1)
        self._slope.set_value(value)

    def bind_with(self, transformer: Self) -> None:
        self._intercept.bind_with(transformer.intercept_parameter)
        self._slope.bind_with(transformer.slope_parameter)

    def forward(self, parameter: torch.Tensor) -> torch.Tensor:

        intercept = cast(torch.Tensor, self._intercept()).expand(
            *parameter.shape
        )
        slope = cast(torch.Tensor, self._slope()).expand(*parameter.shape)

        linear_transformed_parameter = parameter * slope
        positive_mask = parameter >= 0
        negative_mask = ~positive_mask

        _parameter_value = torch.empty_like(parameter, device=parameter.device)
        _parameter_value[positive_mask] = (
            intercept[positive_mask]
            + linear_transformed_parameter[positive_mask]
        )
        _parameter_value[negative_mask] = intercept[negative_mask] * torch.exp(
            linear_transformed_parameter[negative_mask]
            / intercept[negative_mask]
        )

        norm: torch.Tensor = torch.norm(
            _parameter_value, p=1, dim=-1, keepdim=True
        )

        parameter_value = _parameter_value / norm
        return parameter_value

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        if value.numel() == 1:
            raise ValueError("value must have more than one element.")
        negative_mask = value < 0
        if negative_mask.any():
            raise ValueError("value must be non-negative.")
        norm = torch.norm(value, p=1, dim=-1)
        first_element = norm.reshape(-1)[0].item()
        reference = torch.full_like(norm, first_element)
        if not (torch.allclose(norm, reference) and first_element == 1):
            raise ValueError(f"values at last axis must sum to 1.")

        zero_mask = value == 0
        log_nonzero_min = torch.log(value[~zero_mask].min())
        log_zero_value = log_nonzero_min - self._config.log_zero_offset
        value = value.masked_fill(zero_mask, torch.exp(log_zero_value))

        with self._intercept.evaluation_mode():
            intercept = cast(torch.Tensor, self._intercept()).expand(
                *value.shape
            )
        with self._slope.evaluation_mode():
            slope = cast(torch.Tensor, self._slope()).expand(*value.shape)

        translated_value = value - intercept
        positive_mask = translated_value >= 0
        negative_mask = ~positive_mask

        parameter_value = torch.empty_like(value, device=value.device)
        parameter_value[positive_mask] = (
            translated_value[positive_mask] / slope[positive_mask]
        )
        parameter_value[negative_mask] = (
            torch.log(value[negative_mask] / intercept[negative_mask])
            * intercept[negative_mask]
            / slope[negative_mask]
        )
        return parameter_value


LinearSoftmaxT = TypeVar(
    "LinearSoftmaxT", bound=Transformer, default=LinearSoftmaxTransformer
)
