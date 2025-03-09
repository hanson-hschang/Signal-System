from typing import Generic, Tuple, TypeVar, Union, cast

import torch
import torch.nn as nn

from ss.utility.learning.parameter import Parameter
from ss.utility.learning.parameter.manifold.config import (
    ManifoldParameterConfig,
)
from ss.utility.learning.parameter.transformer import Transformer
from ss.utility.learning.parameter.transformer.config import TransformerConfig
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


MPC = TypeVar("MPC", bound=ManifoldParameterConfig)
T = TypeVar("T", bound=Transformer)


class ManifoldParameter(Parameter[MPC], Generic[T, MPC]):
    def __init__(
        self,
        config: MPC,
        shape: Tuple[int, ...],
    ) -> None:
        super().__init__(config, shape)
        self._transformer: T = self._init_transformer()

    def _init_transformer(self) -> T:
        raise NotImplementedError

    @property
    def transformer(self) -> T:
        return self._transformer

    def bind_with(
        self, parameter: Union[nn.Parameter, Parameter, "ManifoldParameter"]
    ) -> None:
        super().bind_with(parameter)
        if isinstance(parameter, ManifoldParameter) and (
            self.__class__ == parameter.__class__
        ):
            self._transformer.bind_with(cast(T, parameter.transformer))

    def forward(self) -> torch.Tensor:
        return self._transformer.forward(super().forward())

    def set_value(self, value: torch.Tensor) -> None:
        with self.evaluation_mode():
            super().set_value(self._transformer.inverse(value))
