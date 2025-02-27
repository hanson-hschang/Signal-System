from typing import assert_never

from dataclasses import dataclass, field
from enum import StrEnum, auto

import torch

from ss.utility.learning.module import config as Config
from ss.utility.learning.module.stochasticizer.config import (
    StochasticizerConfig,
)


@dataclass
class TransitionMatrixConfig(Config.BaseLearningConfig):

    class Option(StrEnum):
        FULL_MATRIX = auto()
        SPATIAL_INVARIANT = auto()
        IID = auto()

    class Initializer(StrEnum):
        NORMAL_DISTRIBUTION = auto()
        UNIFORM_DISTRIBUTION = auto()
        IDENTITY = auto()

        def __init__(self, value: str) -> None:
            self.mean: float = 0.0
            self.variance: float = 1.0
            self.min_value: float = 0.0
            self.max_value: float = 1.0
            self.log_zero_value: float = -10.0

        def initialize(self, dim: int, row_index: int = 0) -> torch.Tensor:
            match self:
                case TransitionMatrixConfig.Initializer.NORMAL_DISTRIBUTION:
                    return torch.normal(
                        self.mean,
                        self.variance,
                        (dim,),
                        # dtype=torch.float64,
                    )
                case TransitionMatrixConfig.Initializer.UNIFORM_DISTRIBUTION:
                    return self.min_value + (
                        self.max_value - self.min_value
                    ) * torch.rand(dim)
                case TransitionMatrixConfig.Initializer.IDENTITY:
                    row = self.log_zero_value * torch.ones(
                        dim,
                        # dtype=torch.float64
                    )
                    row[row_index] = 0.0
                    return torch.Tensor(row)
                case _ as _invalid_initializer:
                    assert_never(_invalid_initializer)

    option: Option = Option.FULL_MATRIX
    initializer: Initializer = Initializer.NORMAL_DISTRIBUTION
    initial_state_binding: bool = False
    stochasticizer: StochasticizerConfig = field(
        default_factory=StochasticizerConfig
    )
    skip_first_transition: bool = False


@dataclass
class TransitionInitialStateConfig(Config.BaseLearningConfig):

    class Initializer(StrEnum):
        NORMAL_DISTRIBUTION = auto()
        UNIFORM_DISTRIBUTION = auto()

        def __init__(self, value: str) -> None:
            self.mean: float = 0.0
            self.variance: float = 1.0
            self.min_value: float = 0.0
            self.max_value: float = 1.0

        def initialize(self, dim: int) -> torch.Tensor:
            match self:
                case (
                    TransitionInitialStateConfig.Initializer.NORMAL_DISTRIBUTION
                ):
                    return torch.normal(
                        self.mean,
                        self.variance,
                        (dim,),
                        # dtype=torch.float64,
                    )
                case (
                    TransitionInitialStateConfig.Initializer.UNIFORM_DISTRIBUTION
                ):
                    return self.min_value + (
                        self.max_value - self.min_value
                    ) * torch.rand(dim)
                case _ as _invalid_initializer:
                    assert_never(_invalid_initializer)

    initializer: Initializer = Initializer.NORMAL_DISTRIBUTION
    stochasticizer: StochasticizerConfig = field(
        default_factory=StochasticizerConfig
    )


@dataclass
class TransitionCoefficientConfig(Config.BaseLearningConfig):
    class Initializer(StrEnum):
        NORMAL_DISTRIBUTION = auto()
        UNIFORM_DISTRIBUTION = auto()

        def __init__(self, value: str) -> None:
            self.mean: float = 0.0
            self.variance: float = 1.0
            self.min_value: float = 0.0
            self.max_value: float = 1.0

        def initialize(self, dim: int) -> torch.Tensor:
            match self:
                case self.NORMAL_DISTRIBUTION:
                    return torch.normal(
                        self.mean,
                        self.variance,
                        (dim,),
                        # dtype=torch.float64,
                    )
                case self.UNIFORM_DISTRIBUTION:
                    return self.min_value + (
                        self.max_value - self.min_value
                    ) * torch.rand(dim)
                case _ as _invalid_initializer:
                    assert_never(_invalid_initializer)  # type: ignore

    initializer: Initializer = Initializer.NORMAL_DISTRIBUTION
    stochasticizer: StochasticizerConfig = field(
        default_factory=StochasticizerConfig
    )


@dataclass
class TransitionConfig(Config.BaseLearningConfig):

    matrix: TransitionMatrixConfig = field(
        default_factory=TransitionMatrixConfig
    )
    initial_state: TransitionInitialStateConfig = field(
        default_factory=TransitionInitialStateConfig
    )
    coefficient: TransitionCoefficientConfig = field(
        default_factory=TransitionCoefficientConfig
    )
