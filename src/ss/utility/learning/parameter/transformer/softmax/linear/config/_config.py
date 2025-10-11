from dataclasses import dataclass, field
from typing import Generic, TypeVar

from ss.utility.assertion.validator import IntegerValidator
from ss.utility.descriptor import DataclassDescriptor
from ss.utility.learning.parameter.initializer import InitializerProtocol
from ss.utility.learning.parameter.initializer.normal_distribution import (
    NormalDistributionInitializer,
)
from ss.utility.learning.parameter.positive.config import (
    PositiveParameterConfig,
)
from ss.utility.learning.parameter.transformer.config import TransformerConfig
from ss.utility.learning.parameter.transformer.exp.config import ExpTC


@dataclass
class InterceptConfig(PositiveParameterConfig[ExpTC], Generic[ExpTC]):
    initializer: InitializerProtocol = field(
        default_factory=lambda: NormalDistributionInitializer.basic_config(
            mean=0.0, std=0.0
        )
    )
    require_training: bool = False

    def __post_init__(self) -> None:
        self.dropout.rate = 0.0


@dataclass
class SlopeConfig(PositiveParameterConfig[ExpTC], Generic[ExpTC]):
    initializer: InitializerProtocol = field(
        default_factory=lambda: NormalDistributionInitializer.basic_config(
            mean=0.0, std=0.0
        )
    )
    require_training: bool = False

    def __post_init__(self) -> None:
        self.dropout.rate = 0.0


@dataclass
class LinearSoftmaxTransformerConfig(TransformerConfig, Generic[ExpTC]):
    class LogZeroOffsetDescriptor(DataclassDescriptor[int]):
        def __set__(self, instance: object, value: int) -> None:
            value = IntegerValidator(value).get_value()
            if value < 0:
                raise ValueError("log_zero_offset must be a positive int.")
            super().__set__(instance, value)

    log_zero_offset: LogZeroOffsetDescriptor = LogZeroOffsetDescriptor(10)
    intercept: InterceptConfig[ExpTC] = field(
        default_factory=lambda: InterceptConfig[ExpTC]()
    )
    slope: SlopeConfig[ExpTC] = field(
        default_factory=lambda: SlopeConfig[ExpTC]()
    )


LinearSoftmaxTC = TypeVar(
    "LinearSoftmaxTC",
    bound=TransformerConfig,
    default=LinearSoftmaxTransformerConfig,
)
