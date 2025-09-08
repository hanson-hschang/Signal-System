from typing import Tuple, Type, override

from dataclasses import dataclass, field

import torch

from ss.utility.assertion.validator import (
    NumberValidator,
    PositiveNumberValidator,
)
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

    # min: MinDescriptor = field(default=MinDescriptor(), init=False, repr=False)
    # max: MaxDescriptor = field(default=MaxDescriptor(), init=False, repr=False)
    min: MinDescriptor = MinDescriptor(0.0)
    max: MaxDescriptor = MaxDescriptor(1.0)

    # def __post_init__(self) -> None:
    #     self._min: float = 0.0
    #     self._max: float = 1.0

    def __call__(self, shape: Tuple[int, ...]) -> torch.Tensor:
        # return self._min + (self._max - self._min) * torch.rand(*shape)
        return self.min + (self.max - self.min) * torch.rand(*shape)

    @classmethod
    @override
    def basic_config(
        cls: Type["UniformDistributionInitializer"],
        *,
        min: float = 0.0,
        max: float = 1.0,
    ) -> "UniformDistributionInitializer":
        initializer = cls()
        initializer.min = min
        initializer.max = max
        return initializer
