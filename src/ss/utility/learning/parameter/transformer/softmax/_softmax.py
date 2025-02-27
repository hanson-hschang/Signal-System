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
        # self._temperature: PP = PositiveParameter(
        #     self._config.temperature, temperature_shape
        # )

    def _init_temperature(self) -> PP:
        temperature_shape = (
            (1,) if len(self._shape) == 1 else self._shape[:-1] + (1,)
        )
        return cast(
            PP, PositiveParameter(self._config.temperature, temperature_shape)
        )

    @property
    def temperature(self) -> PP:
        return self._temperature

    def forward(self, parameter: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(
            parameter / self._temperature(), dim=-1
        )

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        parameter_value: torch.Tensor = torch.log(value) * self._temperature()
        return parameter_value
