from dataclasses import dataclass

from ss.utility.assertion.validator import PositiveIntegerValidator
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

    order: OrderDescriptor = OrderDescriptor(1)
