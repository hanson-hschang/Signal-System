from typing import Callable, Protocol, Sequence, Tuple, TypeVar

import torch
from torch import nn

from ss.utility.learning.module import BaseLearningModule
from ss.utility.learning.module.dropout import Dropout
from ss.utility.learning.parameter import config as Config
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


PC = TypeVar("PC", bound=Config.ParameterConfig)


class Parameter(BaseLearningModule[PC]):

    def __init__(
        self,
        config: PC,
        shape: Tuple[int, ...],
    ) -> None:
        super().__init__(config)
        self._shape = shape
        self._initializer = self._config.initializer
        self._dropout = Dropout(self._config.dropout)
        self._parameter = nn.Parameter(
            self._initializer(self._shape),
            requires_grad=self._config.require_training,
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def parameter(self) -> nn.Parameter:
        return self._parameter

    def binding(self, parameter: nn.Parameter) -> None:
        if not self._parameter.shape == parameter.shape:
            logger.error(
                f"Parameter binding shape mismatch. "
                f"Expected: {self._parameter.shape}. "
                f"Given: {parameter.shape}."
            )
        self._parameter = parameter

    def forward(self) -> torch.Tensor:
        value: torch.Tensor = self._dropout(self._parameter)
        return value

    def set_value(self, value: torch.Tensor) -> None:
        if not self._parameter.shape == value.shape:
            logger.error(
                f"shape mismatch. "
                f"Expected: {self._parameter.shape}. "
                f"Given: {value.shape}."
            )
        with self.evaluation_mode():
            self._parameter.copy_(value)
