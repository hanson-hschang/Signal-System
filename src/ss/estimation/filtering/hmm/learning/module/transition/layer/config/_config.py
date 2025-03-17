from typing import Generic, List, Optional, Tuple, TypeVar

from dataclasses import dataclass, field
from enum import StrEnum, auto

from ss.estimation.filtering.hmm.learning.module.transition.block.config import (
    TransitionBlockConfig,
)
from ss.estimation.filtering.hmm.learning.module.transition.step.config import (
    TransitionStepConfig,
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
    DataclassDescriptor[Tuple[TransitionBlockConfig[TC], ...]], Generic[TC]
):

    def __set__(
        self,
        obj: object,
        value: Tuple[TransitionBlockConfig[TC], ...],
    ) -> None:
        for block in value:
            assert isinstance(block, TransitionBlockConfig), (
                f"Each element of 'blocks' must be of type: 'TransitionBlockConfig'. "
                f"An element given is of type {type(block)}."
            )
        super().__set__(obj, value)


@dataclass
class TransitionLayerConfig(BaseLearningConfig, Generic[TC]):

    blocks: BlocksDescriptor[TC] = BlocksDescriptor[TC](tuple())
    coefficient: TransitionCoefficientConfig[TC] = field(
        default_factory=TransitionCoefficientConfig[TC]
    )
    step: TransitionStepConfig[TC] = field(
        default_factory=TransitionStepConfig[TC]
    )
    block_state_binding: bool = True

    @property
    def block_dim(self) -> int:
        return len(self.blocks)
