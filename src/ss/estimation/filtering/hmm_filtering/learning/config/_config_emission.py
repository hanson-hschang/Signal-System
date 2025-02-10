from dataclasses import dataclass, field
from enum import StrEnum

from ss.learning import BaseLearningConfig


@dataclass
class EmissionMatrixConfig(BaseLearningConfig):

    class Option(StrEnum):
        FULL_MATRIX = "FULL_MATRIX"

    class Initializer(StrEnum):
        NORMAL_DISTRIBUTION = "NORMAL_DISTRIBUTION"
        UNIFORM_DISTRIBUTION = "UNIFORM_DISTRIBUTION"
        CONSTANT = "CONSTANT"

    option: Option = Option.FULL_MATRIX
    initializer: Initializer = Initializer.NORMAL_DISTRIBUTION


@dataclass
class EmissionConfig(BaseLearningConfig):

    matrix: EmissionMatrixConfig = field(default_factory=EmissionMatrixConfig)
