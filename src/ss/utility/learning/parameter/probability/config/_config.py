from typing import Generic, Type, assert_never, cast

from dataclasses import dataclass, field
from enum import StrEnum, auto

from ss.utility.learning.parameter.manifold.config import (
    ManifoldParameterConfig,
)
from ss.utility.learning.parameter.transformer.norm.min_zero.config import (
    MinZeroNormTransformerConfig,
)
from ss.utility.learning.parameter.transformer.softmax.config import (
    SoftmaxTC,
    SoftmaxTransformerConfig,
)
from ss.utility.learning.parameter.transformer.softmax.linear.config import (
    LinearSoftmaxTransformerConfig,
)


@dataclass
class ProbabilityParameterConfig(
    ManifoldParameterConfig[SoftmaxTC], Generic[SoftmaxTC]
):

    class Option(StrEnum):
        SOFTMAX = auto()
        MIN_ZERO_NORM = auto()
        LINEAR_SOFTMAX = auto()

    transformer: SoftmaxTC = field(
        default_factory=lambda: cast(SoftmaxTC, SoftmaxTransformerConfig())
    )

    @classmethod
    def from_option(
        cls, option: Option
    ) -> "ProbabilityParameterConfig[SoftmaxTC]":
        match option:
            case cls.Option.SOFTMAX:
                return cls(
                    transformer=cast(SoftmaxTC, SoftmaxTransformerConfig())
                )
            case cls.Option.MIN_ZERO_NORM:
                return cls(
                    transformer=cast(SoftmaxTC, MinZeroNormTransformerConfig())
                )
            case cls.Option.LINEAR_SOFTMAX:
                return cls(
                    transformer=cast(
                        SoftmaxTC, LinearSoftmaxTransformerConfig()
                    )
                )
            case _ as _probability_option:
                assert_never(_probability_option)

    def get_transformer_type(self) -> Type[SoftmaxTC]:
        return type(self.transformer)
