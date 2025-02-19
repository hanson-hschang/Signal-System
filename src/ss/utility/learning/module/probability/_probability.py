from typing import Literal, Self, Type, assert_never, cast

from functools import partial

import torch
from torch import nn

from ss.utility.learning.module import BaseLearningModule
from ss.utility.learning.module.probability import config as Config
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class Probability(BaseLearningModule[Config.ProbabilityConfig]):
    """
    Probability module.
    """

    def __init__(
        self,
        config: Config.ProbabilityConfig,
    ) -> None:
        super().__init__(config)

    @classmethod
    def create(
        cls,
        config: Config.ProbabilityConfig,
    ) -> "Probability":
        """
        Create a probability module.

        Parameters
        ----------
        config : Config.ProbabilityConfig
            The configuration of the probability module.

        Returns
        -------
        Probability
            The probability module.
        """
        match config.option:
            case Config.ProbabilityConfig.Option.SOFTMAX:
                return SoftmaxProbability(config)

    def get_parameter(self, name: str) -> nn.Parameter:
        """
        Get the parameter of the probability module.

        Parameters
        ----------
        name : str
            The name of the parameter.

        Returns
        -------
        parameter : nn.Parameter
            The parameter.
        """
        match name:
            case "temperature":
                if (
                    not self.config.option
                    == Config.ProbabilityConfig.Option.SOFTMAX
                ):
                    logger.error(
                        f"temperature is not used for the option {self.config.option}."
                    )
                return cast(SoftmaxProbability, self).temperature_parameter
            case _:
                return super().get_parameter(name)


class SoftmaxProbability(Probability):
    def __init__(self, config: Config.ProbabilityConfig) -> None:
        super().__init__(config)
        self._temperature_parameter = nn.Parameter(
            torch.log(torch.tensor(self._config.temperature.initial_value)),
            requires_grad=self._config.temperature.require_training,
        )

    @property
    def temperature_parameter(self) -> nn.Parameter:
        return self._temperature_parameter

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self._temperature_parameter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.softmax(x / self.temperature, dim=-1)
