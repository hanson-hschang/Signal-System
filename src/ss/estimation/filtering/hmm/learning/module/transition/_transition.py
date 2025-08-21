from typing import Generic, List, Tuple, cast

import torch
from torch import nn

from ss.estimation.filtering.hmm.learning.module.filter.config import (
    FilterConfig,
)
from ss.estimation.filtering.hmm.learning.module.transition.config import (
    TransitionConfig,
)
from ss.utility.descriptor import BatchTensorReadOnlyDescriptor
from ss.utility.learning.module import BaseLearningModule, reset_module

# from ss.estimation.filtering.hmm.learning.module.transition.layer import (
#     TransitionLayer,
# )
from ss.utility.learning.parameter.probability import ProbabilityParameter
from ss.utility.learning.parameter.transformer import T
from ss.utility.learning.parameter.transformer.config import TC
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class TransitionModule(
    BaseLearningModule[TransitionConfig[TC]], Generic[T, TC]
):
    def __init__(
        self,
        config: TransitionConfig[TC],
        filter_config: FilterConfig,
    ) -> None:
        super().__init__(config)
        self._state_dim = filter_config.state_dim
        self._initial_state: ProbabilityParameter[
            T, TC
        ] = ProbabilityParameter[T, TC](
            self._config.initial_state.probability_parameter,
            (self._state_dim,),
        )
        self._matrix = ProbabilityParameter[T, TC](
            self._config.matrix.probability_parameter,
            (self._state_dim, self._state_dim),
        )

        # The initial emission layer is counted as a layer with layer_id = 0
        # self._layer_dim = self._config.layer_dim + 1
        # self._layers = nn.ModuleList()
        # for l in range(self._layer_dim - 1):
        #     layer_config = self._config.layers[l]
        #     layer_config.step.skip_first = self._config.skip_first_transition
        #     self._layers.append(
        #         TransitionLayer[T, TC](
        #             layer_config,
        #             filter_config,
        #             l + 1,
        #         )
        #     )

        self._batch_size: int
        with self.evaluation_mode():
            self._init_batch_size(batch_size=1)

    def _init_batch_size(
        self, batch_size: int, is_initialized: bool = False
    ) -> None:
        self._is_initialized = is_initialized
        self._batch_size = batch_size
        # with torch.no_grad():
        #     self._predicted_state_over_layers: torch.Tensor = torch.zeros(
        #         (self._batch_size, self._layer_dim, self._state_dim),
        #     )

    def _check_batch_size(self, batch_size: int) -> None:
        if self._is_initialized:
            assert batch_size == self._batch_size, (
                f"batch_size must be the same as the initialized batch_size. "
                f"batch_size given is {batch_size} while the initialized batch_size is {self._batch_size}."
            )
            return
        self._init_batch_size(batch_size, is_initialized=True)
        self._estimated_state = self.initial_state.repeat(batch_size, 1)

    def _get_internal_estimated_state(
        self, batch_size: int = 1
    ) -> torch.Tensor:
        # if not self._inference:
        #     self._is_initialized = False
        #     estimated_state = self.initial_state.repeat(batch_size, 1)
        #     return estimated_state

        if not self._is_initialized:
            self._is_initialized = True
            self._estimated_state = self.initial_state.repeat(batch_size, 1)
        return self._estimated_state

    def _set_internal_estimated_state(
        self, estimated_state: torch.Tensor
    ) -> None:
        self._estimated_state = estimated_state

    @property
    def initial_state_parameter(
        self,
    ) -> ProbabilityParameter[T, TC]:
        return self._initial_state

    @property
    def initial_state(self) -> torch.Tensor:
        initial_state: torch.Tensor = self._initial_state()
        return initial_state

    @initial_state.setter
    def initial_state(self, initial_state: torch.Tensor) -> None:
        self._initial_state.set_value(initial_state)

    @property
    def matrix_parameter(
        self,
    ) -> ProbabilityParameter[T, TC]:
        return self._matrix

    @property
    def matrix(self) -> torch.Tensor:
        matrix: torch.Tensor = self._matrix()
        return matrix

    @matrix.setter
    def matrix(self, matrix: torch.Tensor) -> None:
        self._matrix.set_value(matrix)

    @property
    def estimated_state(self) -> torch.Tensor:
        batch_size, _ = self._estimated_state.shape
        if batch_size == 1:
            return self._estimated_state.squeeze(0)
        return self._estimated_state

    @property
    def predicted_next_state(self) -> torch.Tensor:
        transition_matrix = self.matrix  # (state_dim, state_dim)
        return self._prediction(self.estimated_state, transition_matrix)

    @torch.compile
    def _prediction(
        self,
        estimated_state: torch.Tensor,
        transition_matrix: torch.Tensor,
    ) -> torch.Tensor:
        predicted_next_state = torch.matmul(estimated_state, transition_matrix)
        return predicted_next_state

    @torch.compile
    def _update(
        self,
        prior_state: torch.Tensor,
        likelihood_state: torch.Tensor,
    ) -> torch.Tensor:
        # update step based on likelihood_state (conditional probability)
        posterior_state = nn.functional.normalize(
            prior_state * likelihood_state,
            p=1,
            dim=1,
        )  # (batch_size, state_dim)
        return posterior_state

    # predicted_state_over_layers = BatchTensorReadOnlyDescriptor(
    #     "_batch_size", "_layer_dim", "_state_dim"
    # )

    # @property
    # def layers(self) -> List[TransitionLayer[T, TC]]:
    #     return [cast(TransitionLayer[T, TC], layer) for layer in self._layers]

    # @property
    # def matrix(self) -> List[torch.Tensor]:
    #     return [
    #         cast(TransitionLayer[T, TC], layer).matrix
    #         for layer in self._layers
    #     ]

    def process(
        self,
        estimated_state: torch.Tensor,
        likelihood_state_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, horizon, _ = likelihood_state_trajectory.shape
        # (batch_size, horizon, state_dim)

        estimated_state_trajectory = torch.empty(
            (batch_size, horizon, self._state_dim),
            device=likelihood_state_trajectory.device,
        )

        # predicted_next_state_trajectory = torch.empty(
        #     (batch_size, horizon, self._state_dim),
        #     device=likelihood_state_trajectory.device,
        # )

        transition_matrix = self.matrix  # (state_dim, state_dim)

        # estimated_state = self._get_internal_estimated_state(
        #     batch_size
        # )  # (batch_size, state_dim)

        # prediction step based on model process (predicted probability)
        # predicted_next_state = self._prediction(
        #     estimated_state,
        #     (
        #         torch.eye(
        #             self._state_dim,
        #             device=likelihood_state_trajectory.device,
        #         )
        #         if self._skip_first_transition
        #         else transition_matrix
        #     ),
        # )  # (batch_size, state_dim)

        # predicted_next_state = (
        #     estimated_state
        #     if self._skip_first_transition
        #     else self._prediction(
        #         estimated_state,
        #         transition_matrix
        #     )
        # )  # (batch_size, state_dim)

        # predicted_next_state = self._first_prediction(
        #     estimated_state, transition_matrix
        # )  # (batch_size, state_dim)

        # print(estimated_state.shape, likelihood_state_trajectory.shape)

        for k in range(horizon):
            likelihood_state = likelihood_state_trajectory[
                :, k, :
            ]  # (batch_size, state_dim)

            # update step based on input_state (conditional probability)
            estimated_state = self._update(
                estimated_state, likelihood_state
            )  # (batch_size, state_dim)

            # prediction step based on model process (predicted probability)
            estimated_state = self._prediction(
                estimated_state, transition_matrix
            )  # (batch_size, state_dim)

            estimated_state_trajectory[:, k, :] = estimated_state
            # predicted_next_state_trajectory[:, k, :] = predicted_next_state.clone()

        # if self._inference:
        #     self._estimated_state = (
        #         predicted_next_state
        #         if self._skip_first_transition
        #         else estimated_state
        #     )

        # if self._inference:
        #     self._set_internal_estimated_state(estimated_state)

        return estimated_state_trajectory

    def forward(self, emission_trajectory: torch.Tensor) -> torch.Tensor:
        if self._inference:
            self._check_batch_size(emission_trajectory.shape[0])
            # Update layer 0 (emission layer) predicted next state result
            # which directly use the emission_trajectory
            # self._predicted_next_state_over_layers[:, 0, :] = (
            #     nn.functional.normalize(
            #         emission_trajectory[
            #             :, -1, :
            #         ],  # (batch_size, horizon=-1, state_dim)
            #         p=1,
            #         dim=1,
            #     )
            # )  # (batch_size, state_dim)
            # self._predicted_state_over_layers[:, 0, :] = emission_trajectory[
            #     :, -1, :
            # ]  # (batch_size, horizon=-1, state_dim)

        # for l, layer in enumerate(self._layers, start=1):
        #     estimated_state_trajectory, predicted_state_trajectory = layer(
        #         emission_trajectory
        #     )  # (batch_size, horizon, state_dim), (batch_size, horizon, state_dim)
        #     if self._inference:
        #         # Update layer l (transition layer l) predicted next state result
        #         self._predicted_state_over_layers[:, l, :] = (
        #             predicted_state_trajectory[:, -1, :]
        #         )  # (batch_size, horizon=-1, state_dim)
        #     emission_trajectory = estimated_state_trajectory
        estimated_state_trajectory = self.process(
            self._get_internal_estimated_state(
                batch_size=emission_trajectory.shape[0]
            ),
            emission_trajectory,
        )

        if self._inference:
            self._set_internal_estimated_state(
                estimated_state_trajectory[:, -1, :]
            )
        else:
            self._is_initialized = False

        return estimated_state_trajectory

    def reset(self) -> None:
        self._is_initialized = False
        self._get_internal_estimated_state()
        self._is_initialized = False
        # for layer in self._layers:
        #     reset_module(layer)
