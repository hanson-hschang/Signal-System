from dataclasses import dataclass
from typing import override

import torch

from ss.utility.assertion.validator import (
    NonnegativeIntegerValidator,
    NumberValidator,
)
from ss.utility.descriptor import DataclassDescriptor
from ss.utility.learning.parameter.initializer import Initializer


@dataclass
class NormalDistributionInitializer(Initializer):
    class MeanDescriptor(DataclassDescriptor[float]):
        def __set__(
            self,
            obj: object,
            value: float,
        ) -> None:
            value = NumberValidator(value).get_value()
            super().__set__(obj, value)

    class StdDescriptor(DataclassDescriptor[float]):
        def __set__(
            self,
            obj: object,
            value: float,
        ) -> None:
            value = NonnegativeIntegerValidator(value).get_value()
            super().__set__(obj, value)

    mean: MeanDescriptor = MeanDescriptor(0.0)
    std: StdDescriptor = StdDescriptor(1.0)

    def __call__(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.normal(self.mean, self.std, shape)

    @classmethod
    @override
    def basic_config(
        cls: type["NormalDistributionInitializer"],
        *,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> "NormalDistributionInitializer":
        initializer = cls()
        initializer.mean = mean
        initializer.std = std
        return initializer
