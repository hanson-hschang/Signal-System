from typing import Optional

import torch

from ss.estimation.filtering.hmm.learning.module import config as Config
from ss.utility.learning.module import BaseLearningModule
from ss.utility.learning.parameter.probability import ProbabilityParameter


class EmissionProcess(BaseLearningModule[Config.EmissionProcessConfig]):
    def __init__(
        self,
        config: Config.EmissionProcessConfig,
        filter_config: Config.FilterConfig,
    ) -> None:
        super().__init__(config)
        self._state_dim = filter_config.state_dim
        self._discrete_observation_dim = filter_config.discrete_observation_dim

        self._matrix = ProbabilityParameter(
            self._config.block.matrix.probability_parameter,
            (self._state_dim, self._discrete_observation_dim),
        )

    @property
    def matrix_parameter(self) -> ProbabilityParameter:
        return self._matrix

    @property
    def matrix(self) -> torch.Tensor:
        matrix: torch.Tensor = self._matrix()
        return matrix

    @matrix.setter
    def matrix(self, matrix: torch.Tensor) -> None:
        self._matrix.set_value(matrix)

    def forward(
        self,
        state_probability_trajectory: torch.Tensor,
        emission_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if emission_matrix is None:
            emission_matrix = self.matrix

        observation_probability_trajectory = torch.matmul(
            state_probability_trajectory,  # (batch_size, horizon, state_dim) or (batch_size, state_dim)
            emission_matrix,  # (state_dim, discrete_observation_dim)
        )  # (batch_size, horizon, discrete_observation_dim) or (batch_size, discrete_observation_dim)

        return observation_probability_trajectory
