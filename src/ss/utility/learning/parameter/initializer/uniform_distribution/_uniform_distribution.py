from typing import Tuple, Type, override

from dataclasses import dataclass, field

import torch

from ss.utility.assertion.validator import (
    NumberValidator,
    PositiveNumberValidator,
)
from ss.utility.descriptor import Descriptor
from ss.utility.learning.parameter.initializer import Initializer


@dataclass
class UniformDistributionInitializer(Initializer):

    class MinDescriptor(Descriptor[float]):
        def __set__(
            self,
            obj: object,
            value: float,
        ) -> None:
            value = NumberValidator(value).get_value()
            super().__set__(obj, value)

    class MaxDescriptor(Descriptor[float]):
        def __set__(
            self,
            obj: object,
            value: float,
        ) -> None:
            value = NumberValidator(value).get_value()
            super().__set__(obj, value)

    min: MinDescriptor = field(default=MinDescriptor(), init=False, repr=False)
    max: MaxDescriptor = field(default=MaxDescriptor(), init=False, repr=False)

    def __post_init__(self) -> None:
        self._min: float = 0.0
        self._max: float = 1.0

    def __call__(self, shape: Tuple[int, ...]) -> torch.Tensor:
        return self._min + (self._max - self._min) * torch.rand(*shape)

    @classmethod
    @override
    def basic_config(  # type: ignore[override]
        cls: Type["UniformDistributionInitializer"],
        min: float = 0.0,
        max: float = 1.0,
    ) -> "UniformDistributionInitializer":
        initializer = cls()
        initializer.min = min
        initializer.max = max
        return initializer
