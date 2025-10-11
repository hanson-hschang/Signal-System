from dataclasses import dataclass, field
from typing import Generic

from ss.utility.learning.module.config import BaseLearningConfig
from ss.utility.learning.parameter.probability.config import (
    ProbabilityParameterConfig,
)
from ss.utility.learning.parameter.transformer.config import TC


@dataclass
class TransitionInitialStateConfig(BaseLearningConfig, Generic[TC]):
    probability_parameter: ProbabilityParameterConfig[TC] = field(
        default_factory=ProbabilityParameterConfig[TC]
    )


@dataclass
class TransitionMatrixConfig(BaseLearningConfig, Generic[TC]):
    probability_parameter: ProbabilityParameterConfig[TC] = field(
        default_factory=ProbabilityParameterConfig[TC]
    )


@dataclass
class TransitionConfig(BaseLearningConfig, Generic[TC]):
    initial_state: TransitionInitialStateConfig[TC] = field(
        default_factory=TransitionInitialStateConfig[TC]
    )
    matrix: TransitionMatrixConfig[TC] = field(
        default_factory=TransitionMatrixConfig[TC]
    )
