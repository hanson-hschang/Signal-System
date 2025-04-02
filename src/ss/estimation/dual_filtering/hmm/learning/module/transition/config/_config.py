from typing import Generic, List, Optional, Tuple, TypeVar

from dataclasses import dataclass, field
from enum import StrEnum, auto

from ss.estimation.dual_filtering.hmm.learning.module.transition.layer.config import (
    DualTransitionLayerConfig,
)
from ss.utility.descriptor import DataclassDescriptor
from ss.utility.learning.module.config import BaseLearningConfig
from ss.utility.learning.parameter.probability.config import (
    ProbabilityParameterConfig,
)
from ss.utility.learning.parameter.transformer.config import TC


class LayersDescriptor(
    DataclassDescriptor[Tuple[DualTransitionLayerConfig[TC], ...]], Generic[TC]
):

    def __set__(
        self,
        obj: object,
        value: Tuple[DualTransitionLayerConfig[TC], ...],
    ) -> None:
        for layer in value:
            assert isinstance(layer, DualTransitionLayerConfig), (
                f"Each element of 'layers' must be of type: 'DualTransitionLayerConfig'. "
                f"An element given is of type {type(layer)}."
            )
        super().__set__(obj, value)


@dataclass
class DualTransitionConfig(BaseLearningConfig, Generic[TC]):

    layers: LayersDescriptor[TC] = LayersDescriptor[TC](tuple())
    skip_first_transition: bool = False

    @property
    def layer_dim(self) -> int:
        return len(self.layers)
