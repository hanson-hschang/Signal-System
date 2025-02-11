from typing import Optional

import torch
from torch import nn

from ss.estimation.filtering.hmm_filtering.learning.config._config import (
    LearningHmmFilterConfig,
)
from ss.learning import BaseLearningModule


class LearningHmmFilterEmissionProcess(
    BaseLearningModule[LearningHmmFilterConfig]
):
    def __init__(
        self,
        config: LearningHmmFilterConfig,
    ) -> None:
        super().__init__(config)
        self._state_dim = self._config.state_dim
        self._discrete_observation_dim = self._config.discrete_observation_dim

        _weight = torch.empty(
            (self._state_dim, self._discrete_observation_dim),
            dtype=torch.float64,
        )
        for i in range(self._state_dim):
            _weight[i, :] = (
                self._config.emission.matrix.initializer.initialize(
                    self._discrete_observation_dim,
                )
            )
        self._weight = nn.Parameter(_weight)

        dropout_rate = (
            self._config.dropout_rate
            if self._discrete_observation_dim > 1
            else 0.0
        )
        self._dropout = nn.Dropout(p=dropout_rate)
        self._mask = torch.ones_like(self._weight)

    @property
    def emission_matrix(self) -> torch.Tensor:
        # mask = torch.empty_like(self._weight).to(dtype=torch.bool)
        # for i in range(self._state_dim):
        #     mask[i] = ~self._dropout(self._mask[i]).to(dtype=torch.bool)
        # weight = nn.functional.softmax(
        #     self._weight.masked_fill(mask, float("-inf")),
        #     dim=1,
        # )
        # weight = nn.functional.softmax(
        #     self._weight,
        #     dim=1,
        # )
        weight = nn.functional.softmax(
            self._dropout(self._weight),
            dim=1,
        )
        return weight

    def forward(
        self,
        state_probability_trajectory: torch.Tensor,
        emission_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if emission_matrix is None:
            emission_matrix = self.emission_matrix

        observation_probability_trajectory = torch.matmul(
            state_probability_trajectory,  # (batch_size, horizon, state_dim) or (batch_size, state_dim)
            emission_matrix,  # (state_dim, discrete_observation_dim)
        )  # (batch_size, horizon, discrete_observation_dim) or (batch_size, discrete_observation_dim)

        return observation_probability_trajectory
