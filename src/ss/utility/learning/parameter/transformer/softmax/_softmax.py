from typing import Generic, Self, TypeVar

import torch

from ss.utility.learning.parameter.positive import PositiveParameter
from ss.utility.learning.parameter.transformer import Transformer
from ss.utility.learning.parameter.transformer.exp import ExpT
from ss.utility.learning.parameter.transformer.exp.config import ExpTC
from ss.utility.learning.parameter.transformer.softmax.config import (
    SoftmaxTransformerConfig,
)


class SoftmaxTransformer(
    Transformer[SoftmaxTransformerConfig[ExpTC]],
    Generic[ExpT, ExpTC],
):
    def __init__(
        self,
        config: SoftmaxTransformerConfig[ExpTC],
        shape: tuple[int, ...],
    ) -> None:
        super().__init__(config, shape)
        self._temperature: PositiveParameter[ExpT, ExpTC] = (
            self._init_temperature()
        )

    def _init_temperature(self) -> PositiveParameter[ExpT, ExpTC]:
        temperature_shape = (
            (1,) if len(self._shape) == 1 else self._shape[:-1] + (1,)
        )
        return PositiveParameter[ExpT, ExpTC](
            self._config.temperature, temperature_shape
        )

    @property
    def temperature_parameter(self) -> PositiveParameter[ExpT, ExpTC]:
        return self._temperature

    @property
    def temperature(self) -> torch.Tensor:
        temperature: torch.Tensor = self._temperature()[..., 0]
        return temperature

    @temperature.setter
    def temperature(self, value: torch.Tensor) -> None:
        if value.ndim == (len(self._shape) - 1):
            value = value.unsqueeze(-1)
        self._temperature.set_value(value)

    def bind_with(self, transformer: Self) -> None:
        self._temperature.bind_with(transformer.temperature_parameter)

    @torch.compile
    def forward(self, parameter: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(
            parameter / self._temperature(), dim=-1
        )

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        # might not be the best way to handle this
        negative_mask = value < 0
        if negative_mask.any():
            raise ValueError("value must be non-negative.")
        zero_mask = value == 0
        log_nonzero_min = torch.log(value[~zero_mask].min())
        log_zero_value = log_nonzero_min - self._config.log_zero_offset
        value = value.masked_fill(zero_mask, torch.exp(log_zero_value))
        with self._temperature.evaluation_mode():
            temperature = self._temperature()
        parameter_value: torch.Tensor = torch.log(value) * temperature
        return parameter_value


SoftmaxT = TypeVar("SoftmaxT", bound=Transformer, default=SoftmaxTransformer)
