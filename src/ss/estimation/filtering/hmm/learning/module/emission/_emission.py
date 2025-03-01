from typing import Optional

import torch

from ss.estimation.filtering.hmm.learning.module import config as Config
from ss.utility.learning.module import BaseLearningModule

# from ss.utility.learning.module.dropout import Dropout
# from ss.utility.learning.module.probability.generator import (
#     ProbabilityGenerator,
# )
from ss.utility.learning.parameter.probability import ProbabilityParameter

# from torch import nn


class LearningHmmFilterEmissionProcess(
    BaseLearningModule[Config.EmissionProcessConfig]
):
    def __init__(
        self,
        config: Config.EmissionProcessConfig,
        filter_config: Config.FilterConfig,
    ) -> None:
        super().__init__(config)
        self._state_dim = filter_config.state_dim
        self._discrete_observation_dim = filter_config.discrete_observation_dim

        # matrix_parameter = torch.empty(
        #     (self._state_dim, self._discrete_observation_dim),
        #     dtype=torch.float64,
        # )
        # for i in range(self._state_dim):
        #     matrix_parameter[i, :] = (
        #         self._config.emission.matrix.initializer(
        #             self._discrete_observation_dim,
        #         )
        #     )

        # self._matrix = self._config.emission.matrix.create(
        #     (self._state_dim, self._discrete_observation_dim),
        # )

        # self._matrix_parameter = nn.Parameter(
        #     self._config.emission.matrix.initializer(
        #         (self._state_dim, self._discrete_observation_dim),
        #     )
        # )

        self._matrix_parameter = ProbabilityParameter(
            self._config.layer.matrix.probability_parameter,
            (self._state_dim, self._discrete_observation_dim),
        )
        # self._matrix_stochasticizer = Stochasticizer.create(
        #     self._config.emission.matrix.stochasticizer
        # )
        # self._probability_generator = ProbabilityGenerator.create(
        #     self._config.emission.matrix.probability.generator
        # )

        # self._config.dropout.rate = (
        #     self._config.dropout.rate
        #     if self._discrete_observation_dim > 1
        #     else 0.0
        # )
        # self._dropout = Dropout(self._config.dropout)
        # self._mask = torch.ones_like(self._weight)

    @property
    def matrix_parameter(self) -> ProbabilityParameter:
        return self._matrix_parameter

    # @property
    # def matrix_stochasticizer(self) -> Stochasticizer:
    #     return self._matrix_stochasticizer

    # @property
    # def probability_generator(self) -> ProbabilityGenerator:
    #     return self._probability_generator

    @property
    def matrix(self) -> torch.Tensor:
        # matrix: torch.Tensor = self._matrix_stochasticizer(
        #     self._dropout(self._matrix_parameter)
        # )
        # matrix: torch.Tensor = self._probability_generator(
        #     self._dropout(self._matrix_parameter)
        # )
        matrix: torch.Tensor = self._matrix_parameter()
        return matrix

        # mask = self._dropout(self._mask).to(device=self._weight.device)
        # extended_weight = torch.cat(
        #     [
        #         self._weight,
        #         torch.ones(
        #             (self._state_dim, 1),
        #             dtype=self._weight.dtype,
        #             device=self._weight.device,
        #         ),
        #     ],
        #     dim=1,
        # )
        # row_norms = (
        #     torch.norm(extended_weight, p=2, dim=1)
        #     .unsqueeze(dim=1)
        #     .expand(self._state_dim, self._discrete_observation_dim)
        # )

        # weight = nn.functional.softmax(
        #     mask * self._weight
        #     + (1 - mask) * row_norms * self._config.dropout.value.log_zero_scale,
        #     dim=1,
        # )
        # return weight

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
