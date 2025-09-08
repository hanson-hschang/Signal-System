from typing import Generic, Optional

from dataclasses import dataclass, field
from enum import StrEnum, auto

from ss.utility.assertion.validator import NonnegativeIntegerValidator
from ss.utility.descriptor import DataclassDescriptor
from ss.utility.learning.module.config import BaseLearningConfig
from ss.utility.learning.parameter.probability.config import (
    ProbabilityParameterConfig,
)
from ss.utility.learning.parameter.transformer.config import TC

# class EstimationDimDescriptor(DataclassDescriptor[int]):
#     def __set__(
#         self,
#         obj: object,
#         value: int,
#     ) -> None:
#         estimation_dim = NonnegativeIntegerValidator(
#             estimation_dim := value
#         ).get_value()
#         super().__set__(obj, estimation_dim)


@dataclass
class EstimationMatrixConfig(BaseLearningConfig, Generic[TC]):

    probability_parameter: ProbabilityParameterConfig[TC] = field(
        default_factory=ProbabilityParameterConfig[TC]
    )


@dataclass
class EstimationConfig(BaseLearningConfig, Generic[TC]):

    class Option(StrEnum):
        ESTIMATED_STATE = auto()
        # PREDICTED_STATE = auto()
        PREDICTED_OBSERVATION_PROBABILITY = auto()
        # PREDICTED_STATE_OVER_LAYERS = auto()
        # PREDICTED_OBSERVATION_PROBABILITY_OVER_LAYERS = auto()
        LINEAR_TRANSFORM_ESTIMATION = auto()
        # LINEAR_TRANSFORM_PREDICTION = auto()

    option: Option = Option.PREDICTED_OBSERVATION_PROBABILITY
    # dim: EstimationDimDescriptor = EstimationDimDescriptor(0)
    matrix: EstimationMatrixConfig[TC] = field(
        default_factory=EstimationMatrixConfig[TC]
    )
