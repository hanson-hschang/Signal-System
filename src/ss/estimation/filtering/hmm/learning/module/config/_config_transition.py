from typing import Generic, List, Optional, Tuple, TypeVar

from dataclasses import dataclass, field
from enum import StrEnum, auto

from ss.utility.descriptor import DataclassDescriptor
from ss.utility.learning.module import config as Config
from ss.utility.learning.parameter.probability.config import (
    ProbabilityParameterConfig,
)
from ss.utility.learning.parameter.transformer.config import TransformerConfig

TC = TypeVar("TC", bound=TransformerConfig)


@dataclass
class TransitionInitialStateConfig(Config.BaseLearningConfig, Generic[TC]):

    probability_parameter: ProbabilityParameterConfig[TC] = field(
        default_factory=lambda: ProbabilityParameterConfig[TC]()
    )


@dataclass
class TransitionMatrixConfig(Config.BaseLearningConfig, Generic[TC]):

    probability_parameter: ProbabilityParameterConfig[TC] = field(
        default_factory=lambda: ProbabilityParameterConfig[TC]()
    )
    initial_state_binding: bool = False


@dataclass
class TransitionBlockConfig(Config.BaseLearningConfig, Generic[TC]):

    class Option(StrEnum):
        FULL_MATRIX = auto()
        SPATIAL_INVARIANT_MATRIX = auto()
        IID = auto()

    option: Option = Option.FULL_MATRIX
    skip_first_transition: bool = False
    matrix: TransitionMatrixConfig[TC] = field(
        default_factory=lambda: TransitionMatrixConfig[TC]()
    )
    initial_state: TransitionInitialStateConfig[TC] = field(
        default_factory=lambda: TransitionInitialStateConfig[TC]()
    )


@dataclass
class TransitionCoefficientConfig(Config.BaseLearningConfig, Generic[TC]):

    probability_parameter: ProbabilityParameterConfig[TC] = field(
        default_factory=lambda: ProbabilityParameterConfig[TC]()
    )


class BlocksDescriptor(
    DataclassDescriptor[Tuple[TransitionBlockConfig[TC], ...]], Generic[TC]
):
    # def __init__(
    #     self, value: Tuple[TransitionBlockConfig[TC]]
    # ) -> None:
    #     if value is None:
    #         value = tuple()
    #     super().__init__(value)

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
class TransitionLayerConfig(Config.BaseLearningConfig, Generic[TC]):

    blocks: BlocksDescriptor[TC] = BlocksDescriptor[TC](tuple())
    coefficient: TransitionCoefficientConfig[TC] = field(
        default_factory=lambda: TransitionCoefficientConfig[TC]()
    )
    initial_state: TransitionInitialStateConfig[TC] = field(
        default_factory=lambda: TransitionInitialStateConfig[TC]()
    )
    block_initial_state_binding: bool = True
    skip_first_transition: bool = False

    @property
    def block_dim(self) -> int:
        return len(self.blocks)


class LayersDescriptor(
    DataclassDescriptor[Tuple[TransitionLayerConfig[TC], ...]], Generic[TC]
):

    # def __init__(
    #     self, value: Optional[Tuple[TransitionLayerConfig[TC]]]
    # ) -> None:
    #     if value is None:
    #         value = field(default_factory=list)
    #     super().__init__(value)

    def __set__(
        self,
        obj: object,
        value: Tuple[TransitionLayerConfig[TC], ...],
    ) -> None:
        for layer in value:
            assert isinstance(layer, TransitionLayerConfig), (
                f"Each element of 'layers' must be of type: 'TransitionLayerConfig'. "
                f"An element given is of type {type(layer)}."
            )
        super().__set__(obj, value)


@dataclass
class TransitionProcessConfig(Config.BaseLearningConfig, Generic[TC]):

    layers: LayersDescriptor[TC] = LayersDescriptor[TC](tuple())
    skip_first_transition: bool = False

    @property
    def layer_dim(self) -> int:
        return len(self.layers)
