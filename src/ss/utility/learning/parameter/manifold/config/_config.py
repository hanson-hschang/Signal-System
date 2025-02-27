from typing import Generic, Type, TypeVar, cast

from dataclasses import dataclass, field

from ss.utility.learning.parameter.config import ParameterConfig
from ss.utility.learning.parameter.transformer import Transformer
from ss.utility.learning.parameter.transformer import config as Config

# T = TypeVar("T", bound=Transformer)
C = TypeVar("C", bound=Config.TransformerConfig)


@dataclass
class ManifoldParameterConfig(ParameterConfig, Generic[C]):
    # Transformer: Type[T] = field(
    #     default_factory=lambda: cast(Type[T], Transformer[TransformerConfig])
    # )
    transformer: C = field(
        default_factory=lambda: cast(C, Config.TransformerConfig())
    )
