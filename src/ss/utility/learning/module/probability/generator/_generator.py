from typing import Sequence, cast

import torch
from torch import nn

from ss.utility.learning.module import BaseLearningModule
from ss.utility.learning.module.probability.generator import config as Config
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class ProbabilityGenerator(
    BaseLearningModule[Config.ProbabilityGeneratorConfig]
):
    """
    ProbabilityGenerator module.
    """

    def __init__(
        self,
        config: Config.ProbabilityGeneratorConfig,
    ) -> None:
        super().__init__(config)
        self._temperature_parameter: nn.Parameter

    @property
    def temperature_parameter(self) -> nn.Parameter:
        if not (
            self._config.option
            is Config.ProbabilityGeneratorConfig.Option.SOFTMAX
        ):
            raise AttributeError(
                f"'temperature_parameter' is only available for the option = {Config.ProbabilityGeneratorConfig.Option.SOFTMAX}."
                f"The current option is {self._config.option}."
            )
        return self._temperature_parameter

    @property
    def temperature(self) -> torch.Tensor:
        if not (
            self._config.option
            is Config.ProbabilityGeneratorConfig.Option.SOFTMAX
        ):
            raise AttributeError(
                f"'temperature' is only available for the option = {Config.ProbabilityGeneratorConfig.Option.SOFTMAX}."
                f"The current option is {self._config.option}."
            )
        return torch.exp(self._temperature_parameter)

    @classmethod
    def create(
        cls,
        config: Config.ProbabilityGeneratorConfig,
        shape: Sequence[int],
    ) -> "ProbabilityGenerator":
        """
        Create a ProbabilityGenerator module.

        Parameters
        ----------
        config : Config.ProbabilityGeneratorConfig
            The configuration of the ProbabilityGenerator module.

        Returns
        -------
        Stochasticizer
            The Stochasticizer module.
        """
        match config.option:
            case Config.ProbabilityGeneratorConfig.Option.SOFTMAX:
                return SoftmaxProbabilityGenerator(config)


class SoftmaxProbabilityGenerator(ProbabilityGenerator):
    def __init__(self, config: Config.ProbabilityGeneratorConfig) -> None:
        super().__init__(config)
        self._temperature_parameter = nn.Parameter(
            torch.log(torch.tensor(self._config.temperature.initial_value)),
            requires_grad=self._config.temperature.require_training,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.softmax(x / self.temperature, dim=-1)
