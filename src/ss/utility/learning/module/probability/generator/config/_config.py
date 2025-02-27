from typing import Any

from dataclasses import dataclass, field
from enum import StrEnum, auto

from ss.utility.assertion.validator import PositiveNumberValidator
from ss.utility.descriptor import ConditionDescriptor, Descriptor
from ss.utility.learning.module.config import BaseLearningConfig
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


@dataclass
class ProbabilityGeneratorConfig(BaseLearningConfig):
    """
    Configuration of the probability generator module.

    Properties
    ----------
    option : Option, default = Option.SOFTMAX
        The option of the probability generator module.
    temperature : TemperatureConfig, only available when option = Option.SOFTMAX
        The configuration of the temperature setting for the softmax probability generator module.
    """

    class Option(StrEnum):
        SOFTMAX = auto()

    option: Option = Option.SOFTMAX

    @dataclass
    class TemperatureConfig(BaseLearningConfig):
        """
        Configuration of the temperature setting for the softmax probability generator module.

        Properties
        ----------
        initial_value : float, default = 1.0, range = (0, ∞)
            The initial value of the temperature.
        learnable : bool, default = False
            Whether the temperature is learnable.
        """

        class InitialValueDescriptor(Descriptor[float]):
            def __set__(self, instance: Any, value: float) -> None:
                value = PositiveNumberValidator(value).get_value()
                super().__set__(instance, value)

        require_training: bool = False

        def __post_init__(self) -> None:
            self._initial_value: float = 1.0

        initial_value: InitialValueDescriptor = field(
            default=InitialValueDescriptor(), init=False
        )

    class TemperatureDescriptor(
        ConditionDescriptor[TemperatureConfig, "ProbabilityGeneratorConfig"]
    ):
        def __get__(
            self,
            obj: "ProbabilityGeneratorConfig",
            obj_type: type,
        ) -> "ProbabilityGeneratorConfig.TemperatureConfig":
            if not (obj.option is ProbabilityGeneratorConfig.Option.SOFTMAX):
                raise AttributeError(
                    f"'temperature' is only available for the option = {obj.Option.SOFTMAX}. "
                    f"The option given is {obj.option}."
                )
            return super().__get__(obj, obj_type)

        def __set__(
            self,
            obj: "ProbabilityGeneratorConfig",
            value: "ProbabilityGeneratorConfig.TemperatureConfig",
        ) -> None:
            if not (obj.option is ProbabilityGeneratorConfig.Option.SOFTMAX):
                raise AttributeError(
                    f"'temperature' is only available for the option = {obj.Option.SOFTMAX}."
                    f"The option given is {obj.option}."
                )
            super().__set__(obj, value)

    def __post_init__(self) -> None:
        self._temperature = ProbabilityGeneratorConfig.TemperatureConfig()

    temperature: TemperatureDescriptor = field(
        default=TemperatureDescriptor(), init=False
    )
