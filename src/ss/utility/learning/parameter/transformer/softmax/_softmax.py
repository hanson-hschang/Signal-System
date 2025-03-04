from typing import Generic, Tuple, TypeVar, cast

import torch

from ss.utility.learning.parameter.positive import PositiveParameter
from ss.utility.learning.parameter.transformer import Transformer
from ss.utility.learning.parameter.transformer.exp import ExpTransformer
from ss.utility.learning.parameter.transformer.softmax import config as Config

T = TypeVar("T", bound=Transformer, default=Transformer)
PP = TypeVar("PP", bound=PositiveParameter, default=PositiveParameter)


class SoftmaxTransformer(
    Transformer[Config.SoftmaxTransformerConfig],
    Generic[PP],
):
    def __init__(
        self,
        config: Config.SoftmaxTransformerConfig,
        shape: Tuple[int, ...],
    ) -> None:
        super().__init__(config, shape)
        self._temperature = self._init_temperature()

    def _init_temperature(self) -> PP:
        temperature_shape = (
            (1,) if len(self._shape) == 1 else self._shape[:-1] + (1,)
        )
        return cast(
            PP, PositiveParameter(self._config.temperature, temperature_shape)
        )

    @property
    def temperature_parameter(self) -> PP:
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
            parameter_value: torch.Tensor = (
                torch.log(value) * self._temperature()
            )
        return parameter_value
