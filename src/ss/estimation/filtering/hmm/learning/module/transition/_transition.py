from typing import List, Tuple, cast

import torch
from torch import nn

from ss.estimation.filtering.hmm.learning.module import config as Config
from ss.utility.descriptor import BatchTensorReadOnlyDescriptor
from ss.utility.learning.module import BaseLearningModule, reset_module
from ss.utility.learning.module.dropout import Dropout
from ss.utility.learning.module.probability import Probability
from ss.utility.logging import Logging

from ._transition_matrix import BaseLearningHmmFilterTransitionMatrix

logger = Logging.get_logger(__name__)


class LearningHmmFilterTransitionLayer(
    BaseLearningModule[Config.LearningHmmFilterConfig]
):
    def __init__(
        self,
        layer_id: int,
        config: Config.LearningHmmFilterConfig,
    ) -> None:
        super().__init__(config)
        self._layer_id = layer_id
        self._feature_dim = self._config.filter.get_feature_dim(
            self._layer_id - 1
        )

        # self._weight_parameter = nn.Parameter(
        #     torch.randn(
        #         self._feature_dim,
        #         dtype=torch.float64,
        #     )
        # )
        self._weight_parameter = nn.Parameter(
            self._config.transition.weight.initializer.initialize(
                self._feature_dim,
            )
        )
        self._weight_probability = Probability.create(
            self._config.transition.weight.probability
        )

        # dropout_rate = (
        #     self._config.dropout.rate if self._feature_dim > 1 else 0.0
        # )
        self._dropout = Dropout(self._config.dropout)
        self._mask = torch.ones_like(self._weight_parameter)

        self._blocks = nn.ModuleList()
        for feature_id in range(self._feature_dim):
            self._blocks.append(
                BaseLearningHmmFilterTransitionMatrix.create(
                    self._config, feature_id
                )
            )

    @property
    def blocks(self) -> List[BaseLearningHmmFilterTransitionMatrix]:
        return [
            cast(BaseLearningHmmFilterTransitionMatrix, block)
            for block in self._blocks
        ]

    @property
    def matrix(self) -> torch.Tensor:
        weight = self._weight_probability(self._weight_parameter)
        transition_matrix = torch.zeros(
            (self._config.filter.state_dim, self._config.filter.state_dim),
            dtype=self._weight_parameter.dtype,
            device=self._weight_parameter.device,
        )
        for i, block in enumerate(self._blocks):
            transition_matrix += (
                cast(BaseLearningHmmFilterTransitionMatrix, block).matrix
                * weight[i]
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
        weight = nn.functional.softmax(
            self._dropout(self._weight_parameter),
            dim=0,
        )

        average_estimated_state_trajectory = torch.zeros_like(
            input_state_trajectory
        )
        average_predicted_next_state_trajectory = torch.zeros_like(
            input_state_trajectory
        )
        for i, block in enumerate(self._blocks):
            estimated_state_trajectory, predicted_next_state_trajectory = (
                block(input_state_trajectory)
            )
            average_estimated_state_trajectory += (
                estimated_state_trajectory * weight[i]
            )
            average_predicted_next_state_trajectory += (
                predicted_next_state_trajectory * weight[i]
            )
        return (
            average_estimated_state_trajectory,
            average_predicted_next_state_trajectory,
        )

    def reset(self) -> None:
        for block in self._blocks:
            reset_module(block)


class LearningHmmFilterTransitionProcess(
    BaseLearningModule[Config.LearningHmmFilterConfig]
):
    def __init__(
        self,
        config: Config.LearningHmmFilterConfig,
    ) -> None:
        super().__init__(config)
        # The initial emission layer is counted as a layer with layer_id = 0
        self._layer_dim = self._config.filter.layer_dim + 1
        self._state_dim = self._config.filter.state_dim
        self._layers = nn.ModuleList()
        for layer_id in range(1, self._layer_dim):
            self._layers.append(
                LearningHmmFilterTransitionLayer(layer_id, self._config)
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
                dtype=torch.float64,
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
            self._predicted_next_state_over_layers[:, 0, :] = (
                nn.functional.normalize(
                    likelihood_state_trajectory[:, -1, :],
                    p=1,
                    dim=1,
                )
            )  # (batch_size, state_dim)

        for i, layer in enumerate(self._layers, start=1):
            estimated_state_trajectory, predicted_next_state_trajectory = (
                layer(likelihood_state_trajectory)
            )
            if self._inference:
                self._predicted_next_state_over_layers[:, i, :] = (
                    predicted_next_state_trajectory[:, -1, :]
                )
            likelihood_state_trajectory = estimated_state_trajectory

        return estimated_state_trajectory, predicted_next_state_trajectory

    def reset(self) -> None:
        self._is_initialized = False
        for layer in self._layers:
            reset_module(layer)
