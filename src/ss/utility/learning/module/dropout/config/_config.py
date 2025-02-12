from dataclasses import dataclass

from ss.utility.learning import config as Config


@dataclass
class DropoutConfig(Config.BaseLearningConfig):
    """
    Configuration of the dropout module.

    Properties
    ----------
    rate : float, default = 0.1
        The dropout rate for the model. (0.0 <= dropout_rate < 1.0)
    """

    rate: float = 0.0
    log_zero_scale: float = -1.0

    def __post_init__(self) -> None:
        assert 0.0 <= self.rate < 1.0, (
            f"dropout_rate must be in the range of [0.0, 1.0). "
            f"dropout_rate given is {self.rate}."
        )
        assert self.log_zero_scale < 0.0, (
            f"log_zero_scale must be less than 0.0. "
            f"log_zero_scale given is {self.log_zero_scale}."
        )
