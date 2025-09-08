from typing import Generic, Tuple, TypeVar, cast

from ss.utility.learning.parameter.manifold import ManifoldParameter
from ss.utility.learning.parameter.probability.config import (
    ProbabilityParameterConfig,
)
from ss.utility.learning.parameter.transformer import T
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
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class ProbabilityParameter(
    ManifoldParameter[T, ProbabilityParameterConfig[TC]], Generic[T, TC]
):
    def __init__(
        self, config: ProbabilityParameterConfig[TC], shape: Tuple[int, ...]
    ) -> None:
        super().__init__(config, shape)

    def _init_transformer(self) -> T:
        transformer_map = {
            SoftmaxTransformerConfig: SoftmaxTransformer,
            MinZeroNormTransformerConfig: MinZeroNormTransformer,
            LinearSoftmaxTransformerConfig: LinearSoftmaxTransformer,
        }

        transformer_class = transformer_map.get(type(self._config.transformer))
        if transformer_class:
            return cast(
                T,
                transformer_class(
                    self._config.transformer,
                    self._shape,
                ),
            )

        raise ValueError(
            f"Unknown transformer config: {self._config.transformer}"
        )
