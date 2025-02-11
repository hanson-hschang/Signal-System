from typing import assert_never

from dataclasses import dataclass, field
from enum import StrEnum

import torch

from ss.learning.config import BaseLearningConfig


@dataclass
class EmissionMatrixConfig(BaseLearningConfig):

    class Option(StrEnum):
        FULL_MATRIX = "FULL_MATRIX"

    class Initializer(StrEnum):
        NORMAL_DISTRIBUTION = "NORMAL_DISTRIBUTION"
        UNIFORM_DISTRIBUTION = "UNIFORM_DISTRIBUTION"

        def __init__(self, value: str) -> None:
            self.mean: float = 0.0
            self.variance: float = 1.0
            self.min_value: float = 0.0
            self.max_value: float = 1.0

        def initialize(self, dim: int) -> torch.Tensor:
            match self:
                case self.NORMAL_DISTRIBUTION:
                    return torch.normal(
                        self.mean,  # type: ignore
                        self.variance,  # type: ignore
                        (dim,),
                        dtype=torch.float64,
                    )
                case self.UNIFORM_DISTRIBUTION:
                    return self.min_value + (  # type: ignore
                        self.max_value - self.min_value  # type: ignore
                    ) * torch.rand(dim, dtype=torch.float64)
                case _ as _invalid_initializer:
                    assert_never(_invalid_initializer)  # type: ignore

    option: Option = Option.FULL_MATRIX
    initializer: Initializer = Initializer.NORMAL_DISTRIBUTION


@dataclass
class EmissionConfig(BaseLearningConfig):

    matrix: EmissionMatrixConfig = field(default_factory=EmissionMatrixConfig)
