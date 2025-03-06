from typing import Generic, Tuple, TypeVar, cast

from ss.utility.learning.parameter.manifold import ManifoldParameter
from ss.utility.learning.parameter.probability import config as Config
from ss.utility.learning.parameter.transformer import Transformer
from ss.utility.learning.parameter.transformer.norm.min_zero import (
    MinZeroNormTransformer,
)
from ss.utility.learning.parameter.transformer.norm.min_zero.config import (
    MinZeroNormTransformerConfig,
)
from ss.utility.learning.parameter.transformer.softmax import (
    SoftmaxTransformer,
)
from ss.utility.learning.parameter.transformer.softmax.config import (
    SoftmaxTransformerConfig,
)

C = TypeVar("C", bound=Config.ProbabilityParameterConfig)
T = TypeVar("T", bound=Transformer, default=SoftmaxTransformer)


class ProbabilityParameter(ManifoldParameter[C, T], Generic[C, T]):
    def __init__(self, config: C, shape: Tuple[int, ...]) -> None:
        super().__init__(config, shape)

    def _init_transformer(self) -> T:
        transformer: Transformer
        if isinstance(self._config.transformer, SoftmaxTransformerConfig):
            transformer = SoftmaxTransformer(
                self._config.transformer, self._shape
            )
        elif isinstance(
            self._config.transformer, MinZeroNormTransformerConfig
        ):
            transformer = MinZeroNormTransformer(
                self._config.transformer, self._shape
            )
        else:
            raise ValueError(
                f"Unknown transformer config: {self._config.transformer}"
            )
        return cast(T, transformer)
