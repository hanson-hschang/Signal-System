from typing import Generic, TypeVar, cast

from ss.utility.learning.parameter.manifold import ManifoldParameter
from ss.utility.learning.parameter.probability import config as Config
from ss.utility.learning.parameter.transformer import Transformer
from ss.utility.learning.parameter.transformer.softmax import (
    SoftmaxTransformer,
)

C = TypeVar("C", bound=Config.ProbabilityParameterConfig)
T = TypeVar("T", bound=Transformer, default=SoftmaxTransformer)


class ProbabilityParameter(ManifoldParameter[C, T], Generic[C, T]):

    def _init_transformer(self) -> T:
        return cast(
            T, SoftmaxTransformer(self._config.transformer, self._shape)
        )
