from typing import Any, Tuple, Type, override

from dataclasses import dataclass, field

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

    # mean: MeanDescriptor = field(
    #     default=MeanDescriptor(), init=False, repr=False
    # )
    # std: StdDescriptor = field(default=StdDescriptor(), init=False, repr=False)
    mean: MeanDescriptor = MeanDescriptor(0.0)
    std: StdDescriptor = StdDescriptor(1.0)

    # def __post_init__(self) -> None:
    #     self._mean: float = 0.0
    #     self._std: float = 1.0

    def __call__(self, shape: Tuple[int, ...]) -> torch.Tensor:
        # return torch.normal(self._mean, self._std, shape)
        return torch.normal(self.mean, self.std, shape)

    @classmethod
    @override
    def basic_config(  # type: ignore[override]
        cls: Type["NormalDistributionInitializer"],
        mean: float = 0.0,
        std: float = 1.0,
    ) -> "NormalDistributionInitializer":
        initializer = cls()
        initializer.mean = mean
        initializer.std = std
        return initializer
