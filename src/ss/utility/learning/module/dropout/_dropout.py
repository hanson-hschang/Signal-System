import torch

from ss.utility.assertion.validator import PositiveNumberValidator
from ss.utility.learning import module as Module
from ss.utility.learning.module.dropout import config as Config


class NoScaleDropout(Module.BaseLearningModule[Config.DropoutConfig]):
    """
    Dropout without rescaling and variable dropout rates.
    """

    def __init__(
        self,
        config: Config.DropoutConfig,
    ) -> None:
        PositiveNumberValidator(rate := config.rate)
        super().__init__(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self._config.rate == 0:
            return x
        else:
            rate = torch.empty(1, device=x.device).uniform_(
                0, self._config.rate
            )
            mask = torch.empty(x.shape, device=x.device).bernoulli_(1 - rate)
            return x * mask
