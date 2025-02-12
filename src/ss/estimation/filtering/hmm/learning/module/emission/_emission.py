from typing import Optional

import torch
from torch import nn

from ss.estimation.filtering.hmm.learning import config as Config
from ss.utility.learning.module import BaseLearningModule
from ss.utility.learning.module.dropout import NoScaleDropout


class LearningHmmFilterEmissionProcess(
    BaseLearningModule[Config.LearningHmmFilterConfig]
):
    def __init__(
        self,
        config: Config.LearningHmmFilterConfig,
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

        self._config.dropout.rate = (
            self._config.dropout.rate
            if self._discrete_observation_dim > 1
            else 0.0
        )
        self._dropout = NoScaleDropout(self._config.dropout)
        self._mask = torch.ones_like(self._weight)

    @property
    def emission_matrix(self) -> torch.Tensor:
        mask = self._dropout(self._mask)
        extended_weight = torch.cat(
            [
                self._weight,
                torch.ones(
                    (self._state_dim, 1),
                    dtype=self._weight.dtype,
                    device=self._weight.device,
                ),
            ],
            dim=1,
        )
        row_norms = (
            torch.norm(extended_weight, p=2, dim=1)
            .unsqueeze(dim=1)
            .expand(self._state_dim, self._discrete_observation_dim)
        )
        weight = nn.functional.softmax(
            mask * self._weight
            + (1 - mask) * row_norms * self._config.dropout.log_zero_scale,
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
