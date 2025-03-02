from dataclasses import dataclass, field
from enum import StrEnum, auto

from ss.utility.learning.module.config import BaseLearningConfig
from ss.utility.learning.parameter.probability.config import (
    ProbabilityParameterConfig,
)


@dataclass
class EmissionMatrixConfig(BaseLearningConfig):

    probability_parameter: ProbabilityParameterConfig = field(
        default_factory=lambda: ProbabilityParameterConfig()
    )


@dataclass
class EmissionBlockConfig(BaseLearningConfig):

    class Option(StrEnum):
        FULL_MATRIX = auto()

    option: Option = Option.FULL_MATRIX
    matrix: EmissionMatrixConfig = field(
        default_factory=lambda: EmissionMatrixConfig()
    )


@dataclass
class EmissionProcessConfig(BaseLearningConfig):

    block: EmissionBlockConfig = field(
        default_factory=lambda: EmissionBlockConfig()
    )
