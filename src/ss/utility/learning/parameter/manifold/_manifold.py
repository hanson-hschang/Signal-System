from typing import Generic, Tuple, TypeVar, cast

import torch

from ss.utility.learning.parameter import Parameter
from ss.utility.learning.parameter.manifold import config as Config
from ss.utility.learning.parameter.transformer import Transformer
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)

# self._transformer: T = self._config.Transformer(
#     self._config.transformer, self._shape
# )

C = TypeVar("C", bound=Config.ManifoldParameterConfig)
T = TypeVar("T", bound=Transformer, default=Transformer)


class ManifoldParameter(Parameter[C], Generic[C, T]):
    def __init__(
        self,
        config: C,
        shape: Tuple[int, ...],
    ) -> None:
        super().__init__(config, shape)
        self._transformer: T = self._init_transformer()

    def _init_transformer(self) -> T:
        raise NotImplementedError
        # return cast(T, Transformer(self._config.transformer, self._shape))

    @property
    def transformer(self) -> T:
        return self._transformer

    def forward(self) -> torch.Tensor:
        return self._transformer.forward(super().forward())

    def set_value(self, value: torch.Tensor) -> None:
        with self.evaluation_mode():
            super().set_value(self._transformer.inverse(value))
