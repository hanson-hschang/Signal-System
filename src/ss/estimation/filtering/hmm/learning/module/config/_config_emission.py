from typing import Callable, Generic, Optional, Self, Tuple, TypeVar

from dataclasses import dataclass, field
from enum import StrEnum, auto

from ss.utility.learning.module.config import BaseLearningConfig
from ss.utility.learning.parameter.probability.config import (
    ProbabilityParameterConfig,
)
from ss.utility.learning.parameter.transformer.config import TransformerConfig

TC = TypeVar("TC", bound=TransformerConfig)


@dataclass
class EmissionMatrixConfig(BaseLearningConfig, Generic[TC]):

    probability_parameter: ProbabilityParameterConfig[TC] = field(
        default_factory=lambda: ProbabilityParameterConfig[TC]()
    )


@dataclass
class EmissionBlockConfig(BaseLearningConfig, Generic[TC]):

    class Option(StrEnum):
        FULL_MATRIX = auto()

    option: Option = Option.FULL_MATRIX
    matrix: EmissionMatrixConfig[TC] = field(
        default_factory=lambda: EmissionMatrixConfig[TC]()
    )


@dataclass
class EmissionProcessConfig(BaseLearningConfig, Generic[TC]):

    block: EmissionBlockConfig[TC] = field(
        default_factory=lambda: EmissionBlockConfig[TC]()
    )
