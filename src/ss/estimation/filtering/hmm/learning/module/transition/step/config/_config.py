from typing import Generic, List, Optional, Tuple, TypeVar

from dataclasses import dataclass, field

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
class TransitionStepConfig(BaseLearningConfig, Generic[TC]):
    initial_state: TransitionInitialStateConfig[TC] = field(
        default_factory=TransitionInitialStateConfig[TC]
    )
    skip_first: bool = False
