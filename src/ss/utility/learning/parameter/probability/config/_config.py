from typing import Generic, TypeVar, assert_never, cast

from dataclasses import dataclass, field
from enum import StrEnum, auto

from ss.utility.learning.parameter.manifold.config import (
    ManifoldParameterConfig,
)
from ss.utility.learning.parameter.transformer.config import TC as TC_SOFTMAX
from ss.utility.learning.parameter.transformer.norm.min_zero.config import (
    MinZeroNormTransformerConfig,
)
from ss.utility.learning.parameter.transformer.softmax.config import (
    SoftmaxTransformerConfig,
)

# TC_SOFTMAX = TypeVar("TC_SOFTMAX", bound=TransformerConfig, default=SoftmaxTransformerConfig)


@dataclass
class ProbabilityParameterConfig(
    ManifoldParameterConfig[TC_SOFTMAX], Generic[TC_SOFTMAX]
):

    class Option(StrEnum):
        SOFTMAX = auto()
        MIN_ZERO_NORM = auto()

    transformer: TC_SOFTMAX = field(
        default_factory=lambda: cast(TC_SOFTMAX, SoftmaxTransformerConfig())
    )

    @classmethod
    def from_option(
        cls, option: Option
    ) -> "ProbabilityParameterConfig[TC_SOFTMAX]":
        match option:
            case cls.Option.SOFTMAX:
                return cls(
                    transformer=cast(TC_SOFTMAX, SoftmaxTransformerConfig())
                )
            case cls.Option.MIN_ZERO_NORM:
                return cls(
                    transformer=cast(
                        TC_SOFTMAX, MinZeroNormTransformerConfig()
                    )
                )
            case _ as _probability_option:
                assert_never(_probability_option)
