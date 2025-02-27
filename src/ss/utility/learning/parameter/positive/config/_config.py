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
C = TypeVar("C", bound=TransformerConfig, default=Config.ExpTransformerConfig)


@dataclass
class PositiveParameterConfig(ManifoldParameterConfig[C], Generic[C]):
    # Transformer: Type[T] = field(
    #     default_factory=lambda: cast(Type[T], ExpTransformer)
    # )
    transformer: C = field(
        default_factory=lambda: cast(C, Config.ExpTransformerConfig())
    )
