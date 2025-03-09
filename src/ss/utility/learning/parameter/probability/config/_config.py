from typing import Generic, TypeVar, assert_never, cast

from dataclasses import dataclass, field
from enum import StrEnum, auto

from ss.utility.learning.parameter.manifold.config import (
    ManifoldParameterConfig,
)
from ss.utility.learning.parameter.transformer.config import TransformerConfig
from ss.utility.learning.parameter.transformer.norm.min_zero.config import (
    MinZeroNormTransformerConfig,
)
from ss.utility.learning.parameter.transformer.softmax.config import (
    SoftmaxTransformerConfig,
)

TC = TypeVar("TC", bound=TransformerConfig, default=SoftmaxTransformerConfig)


@dataclass
class ProbabilityParameterConfig(ManifoldParameterConfig[TC], Generic[TC]):

    class Option(StrEnum):
        SOFTMAX = auto()
        MIN_ZERO_NORM = auto()

    transformer: TC = field(
        default_factory=lambda: cast(TC, SoftmaxTransformerConfig())
    )

    @classmethod
    def from_option(cls, option: Option) -> "ProbabilityParameterConfig[TC]":
        match option:
            case cls.Option.SOFTMAX:
                return cls(transformer=cast(TC, SoftmaxTransformerConfig()))
            case cls.Option.MIN_ZERO_NORM:
                return cls(
                    transformer=cast(TC, MinZeroNormTransformerConfig())
                )
            case _ as _probability_option:
                assert_never(_probability_option)
