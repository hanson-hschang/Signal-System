from typing import Generic, Self, Type, TypeVar, cast

from dataclasses import dataclass, field

from ss.utility.learning.parameter.config import ParameterConfig
from ss.utility.learning.parameter.transformer import Transformer
from ss.utility.learning.parameter.transformer.config import TransformerConfig

# T = TypeVar("T", bound=Transformer)
TC = TypeVar("TC", bound=TransformerConfig)


@dataclass
class ManifoldParameterConfig(ParameterConfig, Generic[TC]):

    transformer: TC = field(
        default_factory=lambda: cast(TC, TransformerConfig())
    )
    # default_factory=lambda: cast(C_contra, Config.TransformerConfig())

    # Transformer: Type[T] = field(
    #     default_factory=lambda: cast(Type[T], Transformer[TransformerConfig])
    # )

    @classmethod
    def create(cls: Type[Self], transformer: TC) -> Self:
        return cls(transformer=transformer)
