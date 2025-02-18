from dataclasses import dataclass
from enum import StrEnum

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
        The dropout rate for the model. (0.0 <= dropout_rate < 1.0)
    """

    class Value(StrEnum):
        ZERO = "ZERO"
        LOG_ZERO = "LOG_ZERO"

        def __init__(self, value: str) -> None:
            self._scaling: bool = True
            self._log_zero_scale: float = -1.0

        @property
        def scaling(self) -> bool:
            if not (self is self.ZERO):
                logger.error(f"scaling is not available for {self}.")
            return self._scaling

        @scaling.setter
        def scaling(self, scaling: bool) -> None:
            if not (self is self.ZERO):
                logger.error(f"scaling is not available for {self}.")
            self._scaling = scaling

        @property
        def log_zero_scale(self) -> float:
            if not (self is self.LOG_ZERO):
                logger.error(f"log_zero_scale is not available for {self}.")
            return self._log_zero_scale

        @log_zero_scale.setter
        def log_zero_scale(self, log_zero_scale: float) -> None:
            if not (self is self.LOG_ZERO):
                logger.error(f"log_zero_scale is not available for {self}.")
            self._log_zero_scale = log_zero_scale

    rate: float = 0.5
    value: Value = Value.ZERO

    def __post_init__(self) -> None:
        assert 0.0 <= self.rate < 1.0, (
            f"dropout_rate must be in the range of [0.0, 1.0). "
            f"dropout_rate given is {self.rate}."
        )
