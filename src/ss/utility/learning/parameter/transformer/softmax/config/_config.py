from typing import Generic, TypeVar, cast

from dataclasses import dataclass, field

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

# TC = TypeVar("TC", bound=TransformerConfig, default=ExpTransformerConfig)


@dataclass
class TemperatureConfig(PositiveParameterConfig[ExpTC], Generic[ExpTC]):
    initializer: InitializerProtocol = field(
        default_factory=lambda: NormalDistributionInitializer.basic_config(
            mean=0.0, std=0.0
        )
    )
    require_training: bool = False

    def __post_init__(self) -> None:
        self.dropout.rate = 0.0


@dataclass
class SoftmaxTransformerConfig(TransformerConfig, Generic[ExpTC]):

    class LogZeroOffsetDescriptor(DataclassDescriptor[int]):
        def __set__(self, instance: object, value: int) -> None:
            value = IntegerValidator(value).get_value()
            if value < 0:
                raise ValueError("log_zero_offset must be a positive int.")
            super().__set__(instance, value)

    log_zero_offset: LogZeroOffsetDescriptor = LogZeroOffsetDescriptor(10)
    temperature: TemperatureConfig[ExpTC] = field(
        default_factory=lambda: TemperatureConfig[ExpTC]()
    )
    # How to remove the cast above? Does not seem right with it...
    # There is one try shown in the following commented out code. It has no error running but has some trouble loading the module.


# Do not do the following. It will need to add operator.getitem and TC(???) as safe globals when loading.
# @dataclass
# class SoftmaxTransformerConfig(TransformerConfig, Generic[TC]):
#     temperature: TemperatureConfig[TC] = field(default_factory=TemperatureConfig[TC])


SoftmaxTC = TypeVar(
    "SoftmaxTC", bound=TransformerConfig, default=SoftmaxTransformerConfig
)
