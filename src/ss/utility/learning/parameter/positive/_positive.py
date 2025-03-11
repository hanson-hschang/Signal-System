from typing import Generic, Tuple, TypeVar, cast

from ss.utility.learning.parameter.manifold import ManifoldParameter
from ss.utility.learning.parameter.positive.config import (
    PositiveParameterConfig,
)
from ss.utility.learning.parameter.transformer import T, Transformer
from ss.utility.learning.parameter.transformer.config import TC
from ss.utility.learning.parameter.transformer.exp import ExpTransformer
from ss.utility.learning.parameter.transformer.exp.config import (
    ExpTransformerConfig,
)

# TC = TypeVar("TC", bound=TransformerConfig)
# T = TypeVar("T", bound=Transformer)


class PositiveParameter(
    ManifoldParameter[T, PositiveParameterConfig[TC]], Generic[T, TC]
):

    def __init__(
        self, config: PositiveParameterConfig[TC], shape: Tuple[int, ...]
    ) -> None:
        super().__init__(config, shape)

    def _init_transformer(self) -> T:
        transformer: Transformer
        if isinstance(self._config.transformer, ExpTransformerConfig):
            transformer = ExpTransformer(self._config.transformer, self._shape)
        else:
            raise ValueError(
                f"Unknown transformer config: {self._config.transformer}"
            )
        return cast(T, transformer)
        # return cast(T, ExpTransformer(self._config.transformer, self._shape))
