from typing import Generic, List, Optional, Tuple, TypeVar

from dataclasses import dataclass, field
from enum import StrEnum, auto

from ss.estimation.filtering.hmm.learning.module.transition.step.config import (
    TransitionStepConfig,
)
from ss.utility.learning.module.config import BaseLearningConfig
from ss.utility.learning.parameter.probability.config import (
    ProbabilityParameterConfig,
)
from ss.utility.learning.parameter.transformer.config import TC


@dataclass
class TransitionMatrixConfig(BaseLearningConfig, Generic[TC]):

    probability_parameter: ProbabilityParameterConfig[TC] = field(
        default_factory=ProbabilityParameterConfig[TC]
    )
    initial_state_binding: bool = False


@dataclass
class TransitionBlockConfig(BaseLearningConfig, Generic[TC]):

    class Option(StrEnum):
        FULL_MATRIX = auto()
        SPATIAL_INVARIANT_MATRIX = auto()
        IID = auto()

    option: Option = Option.FULL_MATRIX
    step: TransitionStepConfig[TC] = field(
        default_factory=TransitionStepConfig[TC]
    )
    # skip_first_transition: bool = False
    matrix: TransitionMatrixConfig[TC] = field(
        default_factory=TransitionMatrixConfig[TC]
    )
    # initial_state: TransitionInitialStateConfig[TC] = field(
    #     default_factory=TransitionInitialStateConfig[TC]
    # )
