from typing import List, Optional

from dataclasses import dataclass, field
from enum import StrEnum, auto

from ss.utility.descriptor import DataclassDescriptor
from ss.utility.learning.module import config as Config
from ss.utility.learning.parameter.probability.config import (
    ProbabilityParameterConfig,
)


@dataclass
class TransitionInitialStateConfig(Config.BaseLearningConfig):

    probability_parameter: ProbabilityParameterConfig = field(
        default_factory=lambda: ProbabilityParameterConfig()
    )


@dataclass
class TransitionMatrixConfig(Config.BaseLearningConfig):

    probability_parameter: ProbabilityParameterConfig = field(
        default_factory=lambda: ProbabilityParameterConfig()
    )
    initial_state_binding: bool = False


@dataclass
class TransitionBlockConfig(Config.BaseLearningConfig):

    class Option(StrEnum):
        FULL_MATRIX = auto()
        SPATIAL_INVARIANT_MATRIX = auto()
        IID = auto()

    option: Option = Option.FULL_MATRIX
    skip_first_transition: bool = False
    matrix: TransitionMatrixConfig = field(
        default_factory=TransitionMatrixConfig
    )
    initial_state: TransitionInitialStateConfig = field(
        default_factory=lambda: TransitionInitialStateConfig()
    )


@dataclass
class TransitionCoefficientConfig(Config.BaseLearningConfig):

    probability_parameter: ProbabilityParameterConfig = field(
        default_factory=lambda: ProbabilityParameterConfig()
    )


@dataclass
class TransitionLayerConfig(Config.BaseLearningConfig):

    class BlocksDescriptor(DataclassDescriptor[List[TransitionBlockConfig]]):
        def __init__(
            self, value: Optional[List[TransitionBlockConfig]] = None
        ) -> None:
            if value is None:
                value = field(default_factory=list)
            super().__init__(value)

        def __set__(
            self,
            obj: object,
            value: List[TransitionBlockConfig],
        ) -> None:
            for layer in value:
                assert isinstance(layer, TransitionBlockConfig), (
                    f"Each element of 'layers' must be of type: 'TransitionLayerConfig'. "
                    f"An element given is of type {type(layer)}."
                )
            super().__set__(obj, value)

    blocks: BlocksDescriptor = BlocksDescriptor()
    coefficient: TransitionCoefficientConfig = field(
        default_factory=lambda: TransitionCoefficientConfig()
    )

    @property
    def block_dim(self) -> int:
        return len(self.blocks)


@dataclass
class TransitionProcessConfig(Config.BaseLearningConfig):

    class LayersDescriptor(DataclassDescriptor[List[TransitionLayerConfig]]):
        def __init__(
            self, value: Optional[List[TransitionLayerConfig]] = None
        ) -> None:
            if value is None:
                value = field(default_factory=list)
            super().__init__(value)

        def __set__(
            self,
            obj: object,
            value: List[TransitionLayerConfig],
        ) -> None:
            for layer in value:
                assert isinstance(layer, TransitionLayerConfig), (
                    f"Each element of 'layers' must be of type: 'TransitionLayerConfig'. "
                    f"An element given is of type {type(layer)}."
                )
            super().__set__(obj, value)

    layers: LayersDescriptor = LayersDescriptor()

    @property
    def layer_dim(self) -> int:
        return len(self.layers)
