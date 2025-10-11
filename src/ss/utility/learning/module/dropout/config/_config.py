from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any

from ss.utility.assertion.validator import NonnegativeNumberValidator
from ss.utility.descriptor import DataclassDescriptor
from ss.utility.learning.module import config as Config
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


@dataclass
class DropoutConfig(Config.BaseLearningConfig):
    """
    Configuration of the dropout module.

    Arguments
    ----------
    rate : float, default = 0.5
        The dropout rate for the model. (0.0 <= rate < 1.0)
    value : Value, default = Value.ZERO
        The value assigned to the dropout tensor element.

    Properties
    ----------
    rate : float
        The dropout rate for the model. (0.0 <= rate < 1.0)
    value : Value
        The value assigned to the dropout tensor element.
    scaling : bool, default = True
        Whether the non-dropout tensor elements are scaled
        when the value is Value.ZERO.
        The scaling value is 1.0 / (1.0 - rate).
    log_zero_scale : float, default = -1.0
        The scaling value for the dropout tensor element
        when the value is Value.LOG_ZERO.
    """

    class RateDescriptor(DataclassDescriptor[float]):
        def __set__(self, instance: Any, value: float) -> None:
            value = NonnegativeNumberValidator(value).get_value()
            if not (value < 1.0):
                raise ValueError(
                    f"dropout_rate must be in the range of [0.0, 1.0). "
                    f"dropout_rate given is {value}."
                )
            super().__set__(instance, value)

    class Value(StrEnum):
        ZERO = auto()
        LOG_ZERO = auto()

    class ScalingDescriptor(DataclassDescriptor[bool, "DropoutConfig"]):
        def _validate_instance(self, instance: "DropoutConfig") -> None:
            if instance.value != DropoutConfig.Value.ZERO:
                raise AttributeError(
                    f"'scaling' is only available for "
                    f"the value = {instance.Value.ZERO}. "
                    f"The value given is {instance.value}."
                )

        def __set__(
            self,
            instance: "DropoutConfig",
            value: bool,
        ) -> None:
            setattr(instance, self.private_name, value)

    class LogZeroScaleDescriptor(DataclassDescriptor[float, "DropoutConfig"]):
        def _validate_instance(self, instance: "DropoutConfig") -> None:
            if instance.value != DropoutConfig.Value.LOG_ZERO:
                raise AttributeError(
                    f"'log_zero_scale' is only available for "
                    f"the value = {instance.Value.LOG_ZERO}. "
                    f"The value given is {instance.value}."
                )

        def __set__(
            self,
            instance: "DropoutConfig",
            value: float,
        ) -> None:
            if not (value < 0.0):
                raise ValueError(
                    f"log_zero_scale must be less than 0.0. "
                    f"log_zero_scale given is {value}."
                )
            setattr(instance, self.private_name, value)

    rate: RateDescriptor = RateDescriptor(0.5)
    value: Value = Value.ZERO
    scaling: ScalingDescriptor = ScalingDescriptor(True)
    log_zero_scale: LogZeroScaleDescriptor = LogZeroScaleDescriptor(-1.0)

    def __str__(self) -> str:
        result = f"DropoutConfig(rate={self.rate}, value={self.value}"
        match self.value:
            case self.Value.ZERO:
                result += f", scaling={self.scaling}"
            case self.Value.LOG_ZERO:
                result += f", log_zero_scale={self.log_zero_scale}"
        result += ")"
        return result

    def __repr__(self) -> str:
        return self.__str__()
