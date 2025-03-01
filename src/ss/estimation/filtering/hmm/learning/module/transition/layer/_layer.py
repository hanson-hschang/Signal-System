from typing import List, Tuple, cast

import torch
from torch import nn

from ss.estimation.filtering.hmm.learning.module import config as Config
from ss.estimation.filtering.hmm.learning.module.transition.block import (
    BaseLearningHmmFilterTransitionBlock,
)
from ss.utility.learning.module import BaseLearningModule, reset_module
from ss.utility.learning.parameter.probability import ProbabilityParameter
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class LearningHmmFilterTransitionLayer(
    BaseLearningModule[Config.TransitionLayerConfig]
):
    def __init__(
        self,
        config: Config.TransitionLayerConfig,
        layer_id: int,
        filter_config: Config.FilterConfig,
    ) -> None:
        super().__init__(config)
        self._state_dim = filter_config.state_dim
        self._layer_id = layer_id
        self._block_dim = self._config.block_dim

        self._coefficient_parameter = ProbabilityParameter(
            self._config.coefficient.probability_parameter,
            (self._state_dim, self._block_dim),
        )

        # self._coefficient_parameter = nn.Parameter(
        #     self._config.transition.coefficient.initializer.initialize(
        #         self._feature_dim,
        #     )
        # )
        # self._coefficient_probability = Stochasticizer.create(
        #     self._config.transition.coefficient.stochasticizer
        # )

        # self._dropout = Dropout(self._config.dropout)
        # self._mask = torch.ones_like(self._coefficient_parameter)

        self._blocks = nn.ModuleList()
        for block_id in range(self._block_dim):
            self._blocks.append(
                BaseLearningHmmFilterTransitionBlock.create(
                    self._config.blocks[block_id], block_id, filter_config
                )
            )

    @property
    def id(self) -> int:
        return self._layer_id

    @property
    def block_dim(self) -> int:
        return self._block_dim

    @property
    def coefficient_parameter(self) -> ProbabilityParameter:
        return self._coefficient_parameter

    @property
    def coefficient(self) -> torch.Tensor:
        coefficient: torch.Tensor = self._coefficient_parameter()
        return coefficient

    @property
    def blocks(self) -> List[BaseLearningHmmFilterTransitionBlock]:
        return [
            cast(BaseLearningHmmFilterTransitionBlock, block)
            for block in self._blocks
        ]

    @property
    def matrix(self) -> torch.Tensor:
        coefficient = self.coefficient
        transition_matrix = torch.zeros(
            (self._state_dim, self._state_dim),
            device=coefficient.device,
        )
        for i, block in enumerate(self._blocks):
            transition_matrix += (
                cast(BaseLearningHmmFilterTransitionBlock, block).matrix
                * coefficient[:, i : i + 1]
            )
        return transition_matrix

    def forward(
        self, input_state_trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # mask = ~self._dropout(self._mask).to(
        #     dtype=torch.bool,
        #     device=self._weight.device,
        # )
        # weight = nn.functional.softmax(
        #     self._weight.masked_fill(mask, float("-inf")),
        #     dim=0,
        # )
        # weight = nn.functional.softmax(
        #     self._dropout(self._coefficient_parameter),
        #     dim=0,
        # )
        coefficient = self.coefficient.unsqueeze(dim=0).unsqueeze(dim=0)
        # (1, 1, state_dim, block_dim)

        average_estimated_state_trajectory = torch.zeros_like(
            input_state_trajectory
        )  # (batch_size, horizon, state_dim)
        average_predicted_next_state_trajectory = torch.zeros_like(
            input_state_trajectory
        )  # (batch_size, horizon, state_dim)
        for i, block in enumerate(self._blocks):
            estimated_state_trajectory, predicted_next_state_trajectory = (
                block(input_state_trajectory)
            )  # (batch_size, horizon, state_dim), (batch_size, horizon, state_dim)
            average_estimated_state_trajectory += (
                estimated_state_trajectory * coefficient[:, :, :, i]
            )
            average_predicted_next_state_trajectory += (
                predicted_next_state_trajectory * coefficient[:, :, :, i]
            )
        return (
            average_estimated_state_trajectory,
            average_predicted_next_state_trajectory,
        )

    def reset(self) -> None:
        for block in self._blocks:
            reset_module(block)
