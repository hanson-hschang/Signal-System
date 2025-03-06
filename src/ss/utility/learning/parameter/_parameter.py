from typing import Callable, Generic, Protocol, Tuple, TypeVar, Union

import torch
from torch import nn

from ss.utility.learning.module import BaseLearningModule
from ss.utility.learning.module.dropout import Dropout
from ss.utility.learning.parameter import config as Config
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


PC = TypeVar("PC", bound=Config.ParameterConfig)


class Parameter(BaseLearningModule[PC], Generic[PC]):

    def __init__(
        self,
        config: PC,
        shape: Tuple[int, ...],
    ) -> None:
        super().__init__(config)
        self._shape = shape
        self._initializer = self._config.initializer
        self._dropout = Dropout(self._config.dropout)
        # self._pytorch_parameter = nn.Parameter(
        #     self._initializer(self._shape),
        #     requires_grad=self._config.require_training,
        # )
        self._pytorch_parameter: nn.Parameter = self._init_parameter()

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def pytorch_parameter(self) -> nn.Parameter:
        return self._pytorch_parameter

    def _init_parameter(self) -> nn.Parameter:
        parameter = nn.Parameter(
            self._initializer(self._shape),
            requires_grad=self._config.require_training,
        )
        return parameter

    def bind_with(self, parameter: Union[nn.Parameter, "Parameter"]) -> None:
        if not self.shape == parameter.shape:
            logger.error(
                f"Parameter binding shape mismatch. "
                f"Expected: {self.shape}. "
                f"Given: {parameter.shape}."
            )
        self._pytorch_parameter = (
            parameter.pytorch_parameter
            if isinstance(parameter, Parameter)
            else parameter
        )

    @staticmethod
    def binding(
        parameter_1: Union[nn.Parameter, "Parameter"],
        parameter_2: Union[nn.Parameter, "Parameter"],
    ) -> None:
        if isinstance(parameter_1, Parameter):
            parameter_1.bind_with(parameter_2)
            return
        parameter_1 = (
            parameter_2.pytorch_parameter
            if isinstance(parameter_2, Parameter)
            else parameter_2
        )
        # if isinstance(parameter_2, Parameter):
        #     parameter_1 = parameter_2.pytorch_parameter
        #     return
        # parameter_1 = parameter_2

    def forward(self) -> torch.Tensor:
        value: torch.Tensor = self._dropout(self._pytorch_parameter)
        return value

    def set_value(self, value: torch.Tensor) -> None:
        if not self.shape == value.shape:
            logger.error(
                f"shape mismatch. "
                f"Expected: {self.shape}. "
                f"Given: {value.shape}."
            )
        with self.evaluation_mode():
            self._pytorch_parameter.copy_(value)
