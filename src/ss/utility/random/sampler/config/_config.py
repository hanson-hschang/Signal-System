from typing import Optional, assert_never

from dataclasses import dataclass, field
from enum import StrEnum, auto

import torch

from ss.utility.assertion.validator import (
    NonnegativeIntegerValidator,
    NonnegativeNumberValidator,
)
from ss.utility.descriptor import DataclassDescriptor
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class TemperatureDescriptor(DataclassDescriptor[float]):
    def __set__(self, instance: object, value: float) -> None:
        value = NonnegativeNumberValidator(value).get_value()
        super().__set__(instance, value)


class MaxNumberOfChoicesDescriptor(DataclassDescriptor[int, "SamplerConfig"]):
    def _validate_instance(self, instance: "SamplerConfig") -> None:
        if instance.option != SamplerConfig.Option.TOP_K:
            raise ValueError(
                "option must be TOP_K to access max_number_of_choices."
            )

    def __set__(self, instance: "SamplerConfig", value: int) -> None:
        value = NonnegativeIntegerValidator(value).get_value()
        setattr(instance, self.private_name, value)


class ProbabilityThresholdDescriptor(
    DataclassDescriptor[float, "SamplerConfig"]
):
    def _validate_instance(self, instance: "SamplerConfig") -> None:
        if instance.option != SamplerConfig.Option.TOP_P:
            raise ValueError(
                "option must be TOP_P to access probability_threshold."
            )

    def __set__(self, instance: "SamplerConfig", value: float) -> None:
        value = NonnegativeNumberValidator(value).get_value()
        if value > 1.0:
            raise ValueError(
                "probability_threshold must be a positive number in between 0 and 1."
            )
        setattr(instance, self.private_name, value)


@dataclass
class SamplerConfig:

    class Option(StrEnum):
        AS_IS = auto()
        TOP_K = auto()
        TOP_P = auto()

    temperature: TemperatureDescriptor = TemperatureDescriptor(1)
    option: Option = Option.AS_IS
    # max_number_of_choices: MaxNumberOfChoicesDescriptor = field(
    #     default_factory=lambda: MaxNumberOfChoicesDescriptor(None)
    # )
    # probability_threshold: ProbabilityThresholdDescriptor = field(
    #     default_factory=lambda: ProbabilityThresholdDescriptor(None)
    # )
    max_number_of_choices: MaxNumberOfChoicesDescriptor = (
        MaxNumberOfChoicesDescriptor(0)
    )
    probability_threshold: ProbabilityThresholdDescriptor = (
        ProbabilityThresholdDescriptor(1.0)
    )

    def __str__(self) -> str:
        result = f"SamplerConfig(temperature={self.temperature}, option={self.option}"

        # Only include max_number_of_choices if relevant
        if self.option == SamplerConfig.Option.TOP_K:
            result += f", max_number_of_choices={self.max_number_of_choices}"

        # Only include probability_threshold if relevant
        if self.option == SamplerConfig.Option.TOP_P:
            result += f", probability_threshold={self.probability_threshold}"

        result += ")"
        return result

    def __repr__(self) -> str:
        return self.__str__()
