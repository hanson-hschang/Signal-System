from typing import Generic, Type, TypeVar, cast

from dataclasses import dataclass, field

from ss.utility.learning.parameter.manifold.config import (
    ManifoldParameterConfig,
)
from ss.utility.learning.parameter.transformer import Transformer
from ss.utility.learning.parameter.transformer.config import TransformerConfig
from ss.utility.learning.parameter.transformer.exp import ExpTransformer
from ss.utility.learning.parameter.transformer.exp import config as Config

# T = TypeVar("T", bound=Transformer, default=ExpTransformer)
TC = TypeVar("TC", bound=TransformerConfig)


@dataclass
class PositiveParameterConfig(ManifoldParameterConfig[TC], Generic[TC]):
    # Transformer: Type[T] = field(
    #     default_factory=lambda: cast(Type[T], ExpTransformer)
    # )
    transformer: TC = field(
        default_factory=lambda: cast(TC, Config.ExpTransformerConfig())
    )
