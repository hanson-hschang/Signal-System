from dataclasses import dataclass

from ss.utility.assertion.validator import (
    PositiveIntegerValidator,
    PositiveNumberValidator,
)
from ss.utility.descriptor import DataclassDescriptor
from ss.utility.learning.parameter.transformer.config import TransformerConfig


@dataclass
class MinZeroNormTransformerConfig(TransformerConfig):

    class OrderDescriptor(DataclassDescriptor[int]):
        def __set__(
            self,
            obj: object,
            value: int,
        ) -> None:
            value = PositiveIntegerValidator(value).get_value()
            super().__set__(obj, value)

    class EvenPowerDescriptor(DataclassDescriptor[int]):
        def __set__(
            self,
            obj: object,
            value: int,
        ) -> None:
            value = PositiveIntegerValidator(value).get_value()
            if not (value % 2 == 0):
                raise ValueError(
                    "even_power must be an even positive integer."
                )
            super().__set__(obj, value)

    class EpsilonDescriptor(DataclassDescriptor[float]):
        def __set__(
            self,
            obj: object,
            value: float,
        ) -> None:
            value = PositiveNumberValidator(value).get_value()
            super().__set__(obj, value)

    order: OrderDescriptor = OrderDescriptor(1)
    even_power: EvenPowerDescriptor = EvenPowerDescriptor(2)
    epsilon: EpsilonDescriptor = EpsilonDescriptor(1e-8)
