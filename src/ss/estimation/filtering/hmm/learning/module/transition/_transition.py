from typing import List, Tuple, cast

import torch
from torch import nn

from ss.estimation.filtering.hmm.learning.module import config as Config
from ss.estimation.filtering.hmm.learning.module.transition.layer import (
    LearningHmmFilterTransitionLayer,
)
from ss.utility.descriptor import BatchTensorReadOnlyDescriptor
from ss.utility.learning.module import BaseLearningModule, reset_module
from ss.utility.learning.module.dropout import Dropout
from ss.utility.learning.module.stochasticizer import Stochasticizer
from ss.utility.learning.parameter.probability import ProbabilityParameter
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


# class LearningHmmFilterTransitionLayer(
#     BaseLearningModule[Config.LearningHmmFilterConfig]
# ):
#     def __init__(
#         self,
#         config: Config.LearningHmmFilterConfig,
#         layer_id: int,
#     ) -> None:
#         super().__init__(config)
#         self._layer_id = layer_id
#         self._block_dim = self._config.filter.get_block_dim(
#             self._layer_id - 1
#         )

#         self._coefficient_parameter = ProbabilityParameter(
#             self._config.transition.coefficient.probability_parameter,
#             (self._config.filter.state_dim, self._block_dim),
#         )

#         # self._coefficient_parameter = nn.Parameter(
#         #     self._config.transition.coefficient.initializer.initialize(
#         #         self._feature_dim,
#         #     )
#         # )
#         # self._coefficient_probability = Stochasticizer.create(
#         #     self._config.transition.coefficient.stochasticizer
#         # )

#         # self._dropout = Dropout(self._config.dropout)
#         # self._mask = torch.ones_like(self._coefficient_parameter)

#         self._blocks = nn.ModuleList()
#         for block_id in range(self._block_dim):
#             self._blocks.append(
#                 BaseLearningHmmFilterTransitionBlock.create(
#                     self._config, block_id
#                 )
#             )

#     @property
#     def id(self) -> int:
#         return self._layer_id

#     @property
#     def block_dim(self) -> int:
#         return self._block_dim

#     @property
#     def coefficient_parameter(self) -> ProbabilityParameter:
#         return self._coefficient_parameter

#     @property
#     def coefficient(self) -> torch.Tensor:
#         coefficient: torch.Tensor = self._coefficient_parameter()
#         return coefficient

#     @property
#     def blocks(self) -> List[BaseLearningHmmFilterTransitionBlock]:
#         return [
#             cast(BaseLearningHmmFilterTransitionBlock, block)
#             for block in self._blocks
#         ]

#     @property
#     def matrix(self) -> torch.Tensor:
#         coefficient = self.coefficient
#         transition_matrix = torch.zeros(
#             (self._config.filter.state_dim, self._config.filter.state_dim),
#             device=coefficient.device,
#         )
#         for i, block in enumerate(self._blocks):
#             transition_matrix += (
#                 cast(BaseLearningHmmFilterTransitionBlock, block).matrix
#                 * coefficient[:, i:i+1]
#             )
#         return transition_matrix

#     def forward(
#         self, input_state_trajectory: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         # mask = ~self._dropout(self._mask).to(
#         #     dtype=torch.bool,
#         #     device=self._weight.device,
#         # )
#         # weight = nn.functional.softmax(
#         #     self._weight.masked_fill(mask, float("-inf")),
#         #     dim=0,
#         # )
#         # weight = nn.functional.softmax(
#         #     self._dropout(self._coefficient_parameter),
#         #     dim=0,
#         # )
#         coefficient = self.coefficient.unsqueeze(dim=0).unsqueeze(dim=0)
#         # (1, 1, state_dim, block_dim)

#         average_estimated_state_trajectory = torch.zeros_like(
#             input_state_trajectory
#         )  # (batch_size, horizon, state_dim)
#         average_predicted_next_state_trajectory = torch.zeros_like(
#             input_state_trajectory
#         )  # (batch_size, horizon, state_dim)
#         for i, block in enumerate(self._blocks):
#             estimated_state_trajectory, predicted_next_state_trajectory = (
#                 block(input_state_trajectory)
#             ) # (batch_size, horizon, state_dim), (batch_size, horizon, state_dim)
#             average_estimated_state_trajectory += (
#                 estimated_state_trajectory * coefficient[:, :, :, i]
#             )
#             average_predicted_next_state_trajectory += (
#                 predicted_next_state_trajectory * coefficient[:, :, :, i]
#             )
#         return (
#             average_estimated_state_trajectory,
#             average_predicted_next_state_trajectory,
#         )

#     def reset(self) -> None:
#         for block in self._blocks:
#             reset_module(block)


class LearningHmmFilterTransitionProcess(
    BaseLearningModule[Config.TransitionProcessConfig]
):
    def __init__(
        self,
        config: Config.TransitionProcessConfig,
        filter_config: Config.FilterConfig,
    ) -> None:
        super().__init__(config)
        self._state_dim = filter_config.state_dim

        # The initial emission layer is counted as a layer with layer_id = 0
        self._layer_dim = self._config.layer_dim + 1
        self._layers = nn.ModuleList()
        for l in range(self._layer_dim - 1):
            self._layers.append(
                LearningHmmFilterTransitionLayer(
                    self._config.layers[l],
                    l + 1,
                    filter_config,
                )
            )
        with self.evaluation_mode():
            self._init_batch_size(batch_size=1)

    def _init_batch_size(
        self, batch_size: int, is_initialized: bool = False
    ) -> None:
        self._is_initialized = is_initialized
        self._batch_size = batch_size
        with torch.no_grad():
            self._predicted_next_state_over_layers: torch.Tensor = torch.zeros(
                (self._batch_size, self._layer_dim, self._state_dim),
            )

    def _check_batch_size(self, batch_size: int) -> None:
        if self._is_initialized:
            assert batch_size == self._batch_size, (
                f"batch_size must be the same as the initialized batch_size. "
                f"batch_size given is {batch_size} while the initialized batch_size is {self._batch_size}."
            )
            return
        self._init_batch_size(batch_size, is_initialized=True)

    predicted_next_state_over_layers = BatchTensorReadOnlyDescriptor(
        "_batch_size", "_layer_dim", "_state_dim"
    )

    @property
    def layers(self) -> List[LearningHmmFilterTransitionLayer]:
        return [
            cast(LearningHmmFilterTransitionLayer, layer)
            for layer in self._layers
        ]

    @property
    def matrix(self) -> List[torch.Tensor]:
        return [
            cast(LearningHmmFilterTransitionLayer, layer).matrix
            for layer in self._layers
        ]

    def forward(
        self, likelihood_state_trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._inference:
            self._check_batch_size(likelihood_state_trajectory.shape[0])
            # Update layer 0 (emission layer) predicted next state result
            # which directly use the normalized likelihood_state_trajectory
            self._predicted_next_state_over_layers[:, 0, :] = (
                nn.functional.normalize(
                    likelihood_state_trajectory[
                        :, -1, :
                    ],  # (batch_size, horizon, state_dim)
                    p=1,
                    dim=1,
                )
            )  # (batch_size, state_dim)

        for i, layer in enumerate(self._layers, start=1):
            estimated_state_trajectory, predicted_next_state_trajectory = (
                layer(likelihood_state_trajectory)
            )
            if self._inference:
                # Update layer i (transition layer i) predicted next state result
                self._predicted_next_state_over_layers[:, i, :] = (
                    predicted_next_state_trajectory[:, -1, :]
                )
            likelihood_state_trajectory = estimated_state_trajectory

        return estimated_state_trajectory, predicted_next_state_trajectory

    def reset(self) -> None:
        self._is_initialized = False
        for layer in self._layers:
            reset_module(layer)
