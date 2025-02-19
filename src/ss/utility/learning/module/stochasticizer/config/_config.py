from dataclasses import dataclass
from enum import StrEnum, auto

from ss.utility.assertion.validator import PositiveNumberValidator
from ss.utility.learning.module import config as Config
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


@dataclass
class StochasticizerConfig(Config.BaseLearningConfig):
    """
    Configuration of the Stochasticizer module.

    Parameters
    ----------
    option : Option, default = Option.SOFTMAX
        The option of the Stochasticizer module.
    """

    class Option(StrEnum):
        SOFTMAX = auto()

    @dataclass
    class TemperatureConfig(Config.BaseLearningConfig):
        """
        Configuration of the temperature setting for the probability module.

        Parameters
        ----------
        initial_value : float, default = 1.0
            The initial value of the temperature.
        learnable : bool, default = False
            Whether the temperature is learnable.
        """

        initial_value: float = 1.0
        require_training: bool = False

        def __post_init__(self) -> None:
            PositiveNumberValidator(self.initial_value)

    option: Option = Option.SOFTMAX

    def __post_init__(self) -> None:
        self._temperature = self.TemperatureConfig()

    @property
    def temperature(self) -> TemperatureConfig:
        if not self.option == self.Option.SOFTMAX:
            logger.error(
                f"temperature is not used for the option {self.option}."
            )
        return self._temperature

    @temperature.setter
    def temperature(self, temperature: TemperatureConfig) -> None:
        self._temperature = temperature
