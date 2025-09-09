from typing import Generic, List, Optional, Tuple, TypeVar

from dataclasses import dataclass, field
from enum import StrEnum, auto

from ss.utility.descriptor import DataclassDescriptor
from ss.utility.learning.module.config import BaseLearningConfig
from ss.utility.learning.parameter.probability.config import (
    ProbabilityParameterConfig,
)
from ss.utility.learning.parameter.transformer.config import TC

# class LayersDescriptor(
#     DataclassDescriptor[Tuple[TransitionLayerConfig[TC], ...]], Generic[TC]
# ):

#     def __set__(
#         self,
#         obj: object,
#         value: Tuple[TransitionLayerConfig[TC], ...],
#     ) -> None:
#         for layer in value:
#             assert isinstance(layer, TransitionLayerConfig), (
#                 f"Each element of 'layers' must be of type: 'TransitionLayerConfig'. "
#                 f"An element given is of type {type(layer)}."
#             )
#         super().__set__(obj, value)


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

    # layers: LayersDescriptor[TC] = LayersDescriptor[TC](tuple())
    # skip_first_transition: bool = False

    # @property
    # def layer_dim(self) -> int:
    #     return len(self.layers)
