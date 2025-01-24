from typing import Optional

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
        self._state_dim = self._config.state_dim
        self._discrete_observation_dim = self._config.discrete_observation_dim

        self._weight = nn.Parameter(
            torch.randn(
                self._state_dim,
                self._discrete_observation_dim,
                dtype=torch.float64,
            )
        )

    @property
    def emission_matrix(self) -> torch.Tensor:
        return nn.functional.softmax(self._weight, dim=1)

    def forward(
        self,
        state_probability_trajectory: torch.Tensor,
        emission_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if emission_matrix is None:
            emission_matrix = self.emission_matrix

        observation_probability_trajectory = torch.matmul(
            state_probability_trajectory,  # (batch_size, horizon, state_dim)
            emission_matrix,  # (state_dim, observation_dim)
        )  # (batch_size, horizon, observation_dim)

        return observation_probability_trajectory
