from typing import Tuple

import torch
from torch import nn

from ss.estimation.filtering.hmm.learning import config as Config
from ss.utility.descriptor import BatchTensorReadOnlyDescriptor
from ss.utility.learning.module import BaseLearningModule, reset_module
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

        self._weight = nn.Parameter(
            torch.randn(
                self._feature_dim,
                dtype=torch.float64,
            )
        )

        dropout_rate = (
            self._config.dropout.rate if self._feature_dim > 1 else 0.0
        )
        self._dropout = nn.Dropout(p=dropout_rate)
        self._mask = torch.ones_like(self._weight)

        self.blocks = nn.ModuleList()
        for feature_id in range(self._feature_dim):
            self.blocks.append(
                BaseLearningHmmFilterTransitionMatrix.create(
                    self._config, feature_id
                )
            )

    def forward(
        self, input_state_trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = ~self._dropout(self._mask).to(
            dtype=torch.bool,
            device=self._weight.device,  # not sure why need this to make self and mask on the same device
        )
        weight = nn.functional.softmax(
            self._weight.masked_fill(mask, float("-inf")),
            dim=0,
        )

        average_estimated_state_trajectory = torch.zeros_like(
            input_state_trajectory
        )
        average_predicted_next_state_trajectory = torch.zeros_like(
            input_state_trajectory
        )
        for i, block in enumerate(self.blocks):
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
        for block in self.blocks:
            reset_module(block)


class LearningHmmFilterTransitionProcess(
    BaseLearningModule[Config.LearningHmmFilterConfig]
):
    def __init__(
        self,
        config: Config.LearningHmmFilterConfig,
    ) -> None:
        super().__init__(config)
        self._layer_dim = self._config.filter.layer_dim + 1
        self._state_dim = self._config.filter.state_dim
        self.layers = nn.ModuleList()
        for layer_id in range(1, self._layer_dim):
            self.layers.append(
                LearningHmmFilterTransitionLayer(layer_id, self._config)
            )
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

        for i, layer in enumerate(self.layers):
            estimated_state_trajectory, predicted_next_state_trajectory = (
                layer(likelihood_state_trajectory)
            )
            if self._inference:
                self._predicted_next_state_over_layers[:, i + 1, :] = (
                    predicted_next_state_trajectory[:, -1, :]
                )
            likelihood_state_trajectory = estimated_state_trajectory

        return estimated_state_trajectory, predicted_next_state_trajectory

    def reset(self) -> None:
        self._is_initialized = False
        for layer in self.layers:
            reset_module(layer)
