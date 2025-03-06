from typing import Any, Tuple

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
    """

    class StateDimDescriptor(DataclassDescriptor[int]):
        def __set__(self, instance: Any, value: int) -> None:
            value = PositiveIntegerValidator(value).get_value()
            super().__set__(instance, value)

    class DiscreteObservationDimDescriptor(DataclassDescriptor[int]):
        def __set__(self, instance: Any, value: int) -> None:
            value = PositiveIntegerValidator(value).get_value()
            super().__set__(instance, value)

    state_dim: StateDimDescriptor = StateDimDescriptor(1)
    discrete_observation_dim: DiscreteObservationDimDescriptor = (
        DiscreteObservationDimDescriptor(1)
    )
