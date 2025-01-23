import torch
from torch import nn

from ss.estimation.filtering.hmm_filtering._hmm_filtering_learning_config import (
    LearningHiddenMarkovModelFilterConfig,
)
from ss.learning import BaseLearningModule


class LearningHiddenMarkovModelFilterEmissionProcess(
    BaseLearningModule[LearningHiddenMarkovModelFilterConfig]
):
    def __init__(
        self,
        config: LearningHiddenMarkovModelFilterConfig,
    ) -> None:
        super().__init__(config)
        self._embedding_dim = self._config.state_dim
        self._input_dim = self._config.discrete_observation_dim

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Linear(
                self._input_dim,
                self._embedding_dim,
                bias=False,
                dtype=torch.float64,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
