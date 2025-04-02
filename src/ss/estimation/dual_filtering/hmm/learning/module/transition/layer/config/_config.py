from typing import Generic, List, Optional, Tuple, TypeVar

from dataclasses import dataclass, field
from enum import StrEnum, auto

from ss.estimation.dual_filtering.hmm.learning.module.transition.block.config import (
    DualTransitionBlockConfig,
)
from ss.estimation.dual_filtering.hmm.learning.module.transition.step.config import (
    DualTransitionStepConfig,
)
from ss.utility.descriptor import DataclassDescriptor
from ss.utility.learning.module.config import BaseLearningConfig
from ss.utility.learning.parameter.probability.config import (
    ProbabilityParameterConfig,
)
from ss.utility.learning.parameter.transformer.config import TC


@dataclass
class TransitionCoefficientConfig(BaseLearningConfig, Generic[TC]):

    probability_parameter: ProbabilityParameterConfig[TC] = field(
        default_factory=ProbabilityParameterConfig[TC]
    )


class BlocksDescriptor(
    DataclassDescriptor[Tuple[DualTransitionBlockConfig[TC], ...]], Generic[TC]
):

    def __set__(
        self,
        obj: object,
        value: Tuple[DualTransitionBlockConfig[TC], ...],
    ) -> None:
        for block in value:
            assert isinstance(block, DualTransitionBlockConfig), (
                f"Each element of 'blocks' must be of type: 'DualTransitionBlockConfig'. "
                f"An element given is of type {type(block)}."
            )
        super().__set__(obj, value)


@dataclass
class DualTransitionLayerConfig(BaseLearningConfig, Generic[TC]):

    blocks: BlocksDescriptor[TC] = BlocksDescriptor[TC](tuple())
    coefficient: TransitionCoefficientConfig[TC] = field(
        default_factory=TransitionCoefficientConfig[TC]
    )
    step: DualTransitionStepConfig[TC] = field(
        default_factory=DualTransitionStepConfig[TC]
    )
    block_state_binding: bool = True

    @property
    def block_dim(self) -> int:
        return len(self.blocks)
