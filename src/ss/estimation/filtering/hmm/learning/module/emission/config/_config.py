from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Generic

from ss.utility.learning.module.config import BaseLearningConfig
from ss.utility.learning.parameter.probability.config import (
    ProbabilityParameterConfig,
)
from ss.utility.learning.parameter.transformer.config import TC


@dataclass
class EmissionMatrixConfig(BaseLearningConfig, Generic[TC]):
    probability_parameter: ProbabilityParameterConfig[TC] = field(
        default_factory=ProbabilityParameterConfig[TC]
    )


# @dataclass
# class EmissionBlockConfig(BaseLearningConfig, Generic[TC]):

#     class Option(StrEnum):
#         FULL_MATRIX = auto()

#     option: Option = Option.FULL_MATRIX
#     matrix: EmissionMatrixConfig[TC] = field(
#         default_factory=EmissionMatrixConfig[TC]
#     )


@dataclass
class ObservationConfig(BaseLearningConfig):
    class Option(StrEnum):
        CATEGORY = auto()
        PROBABILITY = auto()

    option: Option = Option.CATEGORY


@dataclass
class EmissionConfig(BaseLearningConfig, Generic[TC]):
    # block: EmissionBlockConfig[TC] = field(
    #     default_factory=EmissionBlockConfig[TC]
    # )
    matrix: EmissionMatrixConfig[TC] = field(
        default_factory=EmissionMatrixConfig[TC]
    )
    observation: ObservationConfig = field(default_factory=ObservationConfig)
