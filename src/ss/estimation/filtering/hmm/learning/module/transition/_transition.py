from typing import Generic, List, Tuple, TypeVar, cast

import torch
from torch import nn

from ss.estimation.filtering.hmm.learning.module import config as Config
from ss.estimation.filtering.hmm.learning.module.transition.layer import (
    TransitionLayer,
)
from ss.utility.descriptor import BatchTensorReadOnlyDescriptor
from ss.utility.learning.module import BaseLearningModule, reset_module
from ss.utility.learning.parameter.transformer import T
from ss.utility.learning.parameter.transformer.config import TC
from ss.utility.learning.parameter.transformer.softmax import (
    SoftmaxTransformer,
)
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)

# TC = TypeVar("TC", bound=TransformerConfig)
# T = TypeVar("T", bound=Transformer)


class TransitionProcess(
    BaseLearningModule[Config.TransitionProcessConfig[TC]], Generic[T, TC]
):
    def __init__(
        self,
        config: Config.TransitionProcessConfig[TC],
        filter_config: Config.FilterConfig,
    ) -> None:
        super().__init__(config)
        self._state_dim = filter_config.state_dim

        # The initial emission layer is counted as a layer with layer_id = 0
        self._layer_dim = self._config.layer_dim + 1
        self._layers = nn.ModuleList()
        for l in range(self._layer_dim - 1):
            layer_config = self._config.layers[l]
            layer_config.skip_first_transition = (
                self._config.skip_first_transition
            )
            self._layers.append(
                TransitionLayer[T, TC](
                    layer_config,
                    filter_config,
                    l + 1,
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
    def layers(self) -> List[TransitionLayer[T, TC]]:
        return [cast(TransitionLayer[T, TC], layer) for layer in self._layers]

    @property
    def matrix(self) -> List[torch.Tensor]:
        return [
            cast(TransitionLayer[T, TC], layer).matrix
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
                    ],  # (batch_size, horizon=-1, state_dim)
                    p=1,
                    dim=1,
                )
            )  # (batch_size, state_dim)

        for l, layer in enumerate(self._layers, start=1):
            estimated_state_trajectory, predicted_next_state_trajectory = (
                layer(likelihood_state_trajectory)
            )  # (batch_size, horizon, state_dim), (batch_size, horizon, state_dim)
            if self._inference:
                # Update layer l (transition layer l) predicted next state result
                self._predicted_next_state_over_layers[:, l, :] = (
                    predicted_next_state_trajectory[:, -1, :]
                )
            likelihood_state_trajectory = estimated_state_trajectory

        return estimated_state_trajectory, predicted_next_state_trajectory

    def reset(self) -> None:
        self._is_initialized = False
        for layer in self._layers:
            reset_module(layer)
