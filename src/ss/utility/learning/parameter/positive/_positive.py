from typing import Generic, TypeVar, cast

from ss.utility.learning.parameter.manifold import ManifoldParameter
from ss.utility.learning.parameter.positive import config as Config
from ss.utility.learning.parameter.transformer import Transformer
from ss.utility.learning.parameter.transformer.exp import ExpTransformer

C = TypeVar(
    "C",
    bound=Config.PositiveParameterConfig,
    default=Config.PositiveParameterConfig,
)
T = TypeVar("T", bound=Transformer, default=ExpTransformer)


class PositiveParameter(ManifoldParameter[C, T], Generic[C, T]):

    def _init_transformer(self) -> T:
        return cast(T, ExpTransformer(self._config.transformer, self._shape))
