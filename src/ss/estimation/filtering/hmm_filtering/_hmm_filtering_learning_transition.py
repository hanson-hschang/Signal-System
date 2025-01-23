import torch
from torch import nn

from ss.estimation.filtering.hmm_filtering._hmm_filtering_learning_config import (
    LearningHiddenMarkovModelFilterConfig,
)
from ss.estimation.filtering.hmm_filtering._hmm_filtering_learning_transition_block import (
    LearningHiddenMarkovModelFilterBlockOption,
)
from ss.learning import BaseLearningModule, reset_module
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class LearningHiddenMarkovModelFilterTransitionLayer(
    BaseLearningModule[LearningHiddenMarkovModelFilterConfig]
):
    def __init__(
        self,
        layer_id: int,
        config: LearningHiddenMarkovModelFilterConfig,
    ) -> None:
        super().__init__(config)
        self._layer_id = layer_id
        self._feature_dim = self._config.get_feature_dim(self._layer_id)
        self._weight = nn.Parameter(
            torch.randn(
                self._feature_dim,
                dtype=torch.float64,
            )
        )
        dropout_rate = (
            self._config.dropout_rate if self._feature_dim > 1 else 0.0
        )
        self._dropout = nn.Dropout(p=dropout_rate)
        self._mask = torch.ones_like(self._weight)
        self.blocks = nn.ModuleList()
        for feature_id in range(self._feature_dim):
            self.blocks.append(
                LearningHiddenMarkovModelFilterBlockOption.get_block(
                    feature_id,
                    self._config.state_dim,
                    self._config.block_option,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = ~self._dropout(self._mask).to(dtype=torch.bool)
        weight = nn.functional.softmax(
            self._weight.masked_fill(mask, float("-inf")),
            dim=0,
        )

        weighted_average = torch.zeros_like(x)
        for i, block in enumerate(self.blocks):
            weighted_average += block(x) * weight[i]
        return weighted_average

    def reset(self) -> None:
        for block in self.blocks:
            reset_module(block)


class LearningHiddenMarkovModelFilterTransitionProcess(
    BaseLearningModule[LearningHiddenMarkovModelFilterConfig]
):
    def __init__(
        self,
        config: LearningHiddenMarkovModelFilterConfig,
    ) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList()
        for layer_id in range(self._config.get_layer_dim()):
            self.layers.append(
                LearningHiddenMarkovModelFilterTransitionLayer(
                    layer_id, self._config
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def reset(self) -> None:
        for layer in self.layers:
            reset_module(layer)
