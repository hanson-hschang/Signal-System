from typing import Generic, Tuple, TypeVar, cast

from ss.utility.learning.parameter.manifold import ManifoldParameter
from ss.utility.learning.parameter.probability.config import (
    ProbabilityParameterConfig,
)
from ss.utility.learning.parameter.transformer import T, Transformer
from ss.utility.learning.parameter.transformer.config import TC
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
from ss.utility.learning.parameter.transformer.softmax.linear import (
    LinearSoftmaxTransformer,
)
from ss.utility.learning.parameter.transformer.softmax.linear.config import (
    LinearSoftmaxTransformerConfig,
)

# TC = TypeVar("TC", bound=TransformerConfig)
# T = TypeVar("T", bound=Transformer)


class ProbabilityParameter(
    ManifoldParameter[T, ProbabilityParameterConfig[TC]], Generic[T, TC]
):
    def __init__(
        self, config: ProbabilityParameterConfig[TC], shape: Tuple[int, ...]
    ) -> None:
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
        elif isinstance(
            self._config.transformer, LinearSoftmaxTransformerConfig
        ):
            transformer = LinearSoftmaxTransformer(
                self._config.transformer, self._shape
            )
        else:
            raise ValueError(
                f"Unknown transformer config: {self._config.transformer}"
            )
        return cast(T, transformer)
