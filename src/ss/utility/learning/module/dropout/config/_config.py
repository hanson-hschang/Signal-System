from typing import Any

from dataclasses import dataclass, field
from enum import StrEnum, auto

from ss.utility.descriptor import ConditionDescriptor, Descriptor
from ss.utility.learning.module import config as Config
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


@dataclass
class DropoutConfig(Config.BaseLearningConfig):
    """
    Configuration of the dropout module.

    Properties
    ----------
    rate : float, default = 0.5
        The dropout rate for the model. (0.0 <= rate < 1.0)
    value : Value, default = Value.ZERO
        The value assigned to the dropout tensor element.
    scaling : bool, default = True
        Whether the non-dropout tensor elements are scaled when the value is Value.ZERO.
        The scaling value is 1.0 / (1.0 - rate).
    log_zero_scale : float, default = -1.0
        The scaling value for the dropout tensor element when the value is Value.LOG_ZERO.
    """

    # def __init__(self, value: str) -> None:
    #     super().__init__()
    #     self._scaling: bool = True
    #     self._log_zero_scale: float = -1.0

    # @property
    # def scaling(self) -> bool:
    #     if not (self == DropoutConfig.Value.ZERO):
    #         logger.error(f"scaling is not available for {self}.")
    #     return self._scaling

    # @scaling.setter
    # def scaling(self, scaling: bool) -> None:
    #     if not (self == DropoutConfig.Value.ZERO):
    #         logger.error(f"scaling is not available for {self}.")
    #     self._scaling = scaling

    # @property
    # def log_zero_scale(self) -> float:
    #     if not (self == DropoutConfig.Value.LOG_ZERO):
    #         logger.error(f"log_zero_scale is not available for {self}.")
    #     return self._log_zero_scale

    # @log_zero_scale.setter
    # def log_zero_scale(self, log_zero_scale: float) -> None:
    #     if not (self == DropoutConfig.Value.LOG_ZERO):
    #         logger.error(f"log_zero_scale is not available for {self}.")
    #     self._log_zero_scale = log_zero_scale

    class RateDescriptor(Descriptor[float]):
        def __set__(self, instance: Any, value: float) -> None:
            if not (0.0 <= value < 1.0):
                logger.error(
                    f"dropout_rate must be in the range of [0.0, 1.0). "
                    f"dropout_rate given is {value}."
                )
            super().__set__(instance, value)

    class Value(StrEnum):
        ZERO = auto()
        LOG_ZERO = auto()

    class ScalingDescriptor(ConditionDescriptor[bool, "DropoutConfig"]):
        def __get__(
            self,
            obj: "DropoutConfig",
            obj_type: type,
        ) -> bool:
            if not (obj.value is DropoutConfig.Value.ZERO):
                raise AttributeError(
                    f"'scaling' is only available for the value = {obj.Value.ZERO}. "
                    f"The value given is {obj.value}."
                )
            return super().__get__(obj, obj_type)

        def __set__(
            self,
            obj: "DropoutConfig",
            value: bool,
        ) -> None:
            if not (obj.value is DropoutConfig.Value.ZERO):
                raise AttributeError(
                    f"'scaling' is only available for the value = {obj.Value.ZERO}. "
                    f"The value given is {obj.value}."
                )
            super().__set__(obj, value)

    class LogZeroScaleDescriptor(ConditionDescriptor[float, "DropoutConfig"]):
        def __get__(
            self,
            obj: "DropoutConfig",
            obj_type: type,
        ) -> float:
            if not (obj.value is DropoutConfig.Value.LOG_ZERO):
                raise AttributeError(
                    f"'log_zero_scale' is only available for the value = {obj.Value.LOG_ZERO}. "
                    f"The value given is {obj.value}."
                )
            return super().__get__(obj, obj_type)

        def __set__(
            self,
            obj: "DropoutConfig",
            value: float,
        ) -> None:
            if not (obj.value is DropoutConfig.Value.LOG_ZERO):
                raise AttributeError(
                    f"'log_zero_scale' is only available for the value = {obj.Value.LOG_ZERO}. "
                    f"The value given is {obj.value}."
                )
            if not (value < 0.0):
                raise ValueError(
                    f"log_zero_scale must be less than 0.0. "
                    f"log_zero_scale given is {value}."
                )
            super().__set__(obj, value)

    value: Value = Value.ZERO
    rate: RateDescriptor = field(default=RateDescriptor(), init=False)
    scaling: ScalingDescriptor = field(default=ScalingDescriptor(), init=False)
    log_zero_scale: LogZeroScaleDescriptor = field(
        default=LogZeroScaleDescriptor(), init=False
    )

    def __post_init__(self) -> None:
        self._rate: float = 0.5
        self._scaling: bool = True
        self._log_zero_scale: float = -1.0
