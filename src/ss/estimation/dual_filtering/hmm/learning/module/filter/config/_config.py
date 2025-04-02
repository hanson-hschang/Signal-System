from typing import Any, Tuple

from dataclasses import dataclass, field

from ss.utility.assertion.validator import (
    NonnegativeIntegerValidator,
    PositiveIntegerValidator,
)
from ss.utility.descriptor import DataclassDescriptor
from ss.utility.learning.module.config import BaseLearningConfig


@dataclass
class DualFilterConfig(BaseLearningConfig):
    """
    Properties
    ----------
    state_dim : int
        The dimension of the state.
    discrete_observation_dim : int
        The dimension of the discrete observation.
    estimation_dim : int
        The dimension of the estimation.
    history_length : int
        The length of the history.
    """

    class StateDimDescriptor(DataclassDescriptor[int]):
        def __set__(self, instance: Any, value: int) -> None:
            value = PositiveIntegerValidator(value).get_value()
            super().__set__(instance, value)

    class DiscreteObservationDimDescriptor(DataclassDescriptor[int]):
        def __set__(self, instance: Any, value: int) -> None:
            value = PositiveIntegerValidator(value).get_value()
            super().__set__(instance, value)

    class EstimationDimDescriptor(DataclassDescriptor[int]):
        def __set__(self, instance: Any, value: int) -> None:
            value = NonnegativeIntegerValidator(value).get_value()
            super().__set__(instance, value)

    class HistoryLengthDescriptor(DataclassDescriptor[int]):
        def __set__(self, instance: Any, value: int) -> None:
            value = PositiveIntegerValidator(value).get_value()
            super().__set__(instance, value)

    state_dim: StateDimDescriptor = StateDimDescriptor(1)
    discrete_observation_dim: DiscreteObservationDimDescriptor = (
        DiscreteObservationDimDescriptor(1)
    )
    estimation_dim: EstimationDimDescriptor = EstimationDimDescriptor(0)
    history_length: HistoryLengthDescriptor = HistoryLengthDescriptor(1)
