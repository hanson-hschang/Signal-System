from dataclasses import dataclass
from typing import override

import torch

from ss.utility.assertion.validator import NumberValidator
from ss.utility.descriptor import DataclassDescriptor
from ss.utility.learning.parameter.initializer import Initializer


@dataclass
class UniformDistributionInitializer(Initializer):
    class MinDescriptor(DataclassDescriptor[float]):
        def __set__(
            self,
            obj: object,
            value: float,
        ) -> None:
            value = NumberValidator(value).get_value()
            super().__set__(obj, value)

    class MaxDescriptor(DataclassDescriptor[float]):
        def __set__(
            self,
            obj: object,
            value: float,
        ) -> None:
            value = NumberValidator(value).get_value()
            super().__set__(obj, value)

    min: MinDescriptor = MinDescriptor(0.0)
    max: MaxDescriptor = MaxDescriptor(1.0)

    def __call__(self, shape: tuple[int, ...]) -> torch.Tensor:
        return self.min + (self.max - self.min) * torch.rand(*shape)

    @classmethod
    @override
    def basic_config(
        cls: type["UniformDistributionInitializer"],
        *,
        min: float = 0.0,
        max: float = 1.0,
    ) -> "UniformDistributionInitializer":
        initializer = cls()
        initializer.min = min
        initializer.max = max
        return initializer
