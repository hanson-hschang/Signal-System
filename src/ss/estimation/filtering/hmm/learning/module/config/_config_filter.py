from typing import Tuple

from dataclasses import dataclass, field

from ss.utility.assertion.validator import PositiveIntegerValidator
from ss.utility.descriptor import DataclassDescriptor
from ss.utility.learning.module import config as Config


@dataclass
class FilterConfig(Config.BaseLearningConfig):
    """

    Properties
    ----------
    state_dim : int
        The dimension of the state.
    discrete_observation_dim : int
        The dimension of the discrete observation.
    block_dim_over_layers : Tuple[int, ...]
        The dimension of blocks for each layer.
        The length of the tuple is the number of layers.
        The values of the tuple (positive integers) are the dimension of blocks for each layer.
    """

    class StateDimDescriptor(DataclassDescriptor[int]):
        def __set__(self, obj: object, value: int) -> None:
            value = PositiveIntegerValidator(value).get_value()
            super().__set__(obj, value)

    class DiscreteObservationDimDescriptor(DataclassDescriptor[int]):
        def __set__(self, obj: object, value: int) -> None:
            value = PositiveIntegerValidator(value).get_value()
            super().__set__(obj, value)

    state_dim: StateDimDescriptor = StateDimDescriptor(1)
    discrete_observation_dim: DiscreteObservationDimDescriptor = (
        DiscreteObservationDimDescriptor(1)
    )
    # block_dim_over_layers: Tuple[int, ...] = field(
    #     default_factory=lambda: (1,)
    # )

    # def __post_init__(self) -> None:
    # self._state_dim: int = 1
    # self._discrete_observation_dim: int = 1
    # self.state_dim = PositiveIntegerValidator(self.state_dim).get_value()
    # self.discrete_observation_dim = PositiveIntegerValidator(
    #     self.discrete_observation_dim
    # ).get_value()
    # for block_dim in self.block_dim_over_layers:
    #     assert type(block_dim) == int, (
    #         f"block_dim_over_layers must be a tuple of integers. "
    #         f"block_dim_over_layers given is {self.block_dim_over_layers}."
    #     )

    # @property
    # def layer_dim(self) -> int:
    #     return len(self.block_dim_over_layers)

    # def get_block_dim(self, layer_id: int) -> int:
    #     return self.block_dim_over_layers[layer_id]
