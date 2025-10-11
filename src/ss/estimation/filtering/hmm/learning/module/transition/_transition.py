from typing import Generic

import torch
from torch import nn

from ss.estimation.filtering.hmm.learning.module.filter.config import (
    FilterConfig,
)
from ss.estimation.filtering.hmm.learning.module.transition.config import (
    TransitionConfig,
)
from ss.utility.descriptor import BatchTensorDescriptor
from ss.utility.learning.module import BaseLearningModule
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
        self._initial_state: ProbabilityParameter[T, TC] = (
            ProbabilityParameter[T, TC](
                self._config.initial_state.probability_parameter,
                (self._state_dim,),
            )
        )
        self._matrix = ProbabilityParameter[T, TC](
            self._config.matrix.probability_parameter,
            (self._state_dim, self._state_dim),
        )

        self._init_batch_size(batch_size=1)

    def _init_batch_size(
        self, batch_size: int, is_initialized: bool = False
    ) -> None:
        self._is_initialized = is_initialized
        self._batch_size = batch_size
        with self.evaluation_mode():
            self._estimated_state = self.initial_state.repeat(
                self._batch_size, 1
            )

    def _check_batch_size(self, batch_size: int) -> None:
        if self._is_initialized:
            assert batch_size == self._batch_size, (
                f"batch_size must be the same as the initialized batch_size. "
                f"batch_size given is {batch_size} while the "
                f"initialized batch_size is {self._batch_size}."
            )
            return
        self._init_batch_size(batch_size, is_initialized=True)

    estimated_state = BatchTensorDescriptor(
        "_batch_size",
        "_state_dim",
    )

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

    def _process(
        self,
        estimated_state: torch.Tensor,
        likelihood_state: torch.Tensor,
    ) -> torch.Tensor:
        # update step based on input_state (conditional probability)
        estimated_state = self._update(
            estimated_state, likelihood_state
        )  # (batch_size, state_dim)

        # prediction step based on model process (predicted probability)
        estimated_state = self._prediction(
            estimated_state, self.matrix
        )  # (batch_size, state_dim)

        return estimated_state

    def forward(self, emission_trajectory: torch.Tensor) -> torch.Tensor:
        batch_size, _, horizon = emission_trajectory.shape
        # (batch_size, state_dim, horizon)

        estimated_state_trajectory = torch.empty(
            (batch_size, self._state_dim, horizon),
            device=emission_trajectory.device,
        )

        estimated_state = self.initial_state.repeat(batch_size, 1)
        # (batch_size, state_dim)

        for k in range(horizon):
            estimated_state = self._process(
                estimated_state,
                emission_trajectory[:, :, k],
            )

            estimated_state_trajectory[:, :, k] = estimated_state

        return estimated_state_trajectory

    @torch.inference_mode()
    def at_inference(self, emission_trajectory: torch.Tensor) -> torch.Tensor:
        batch_size, _, horizon = emission_trajectory.shape

        self._check_batch_size(batch_size)

        for k in range(horizon):
            self._estimated_state = self._process(
                self._estimated_state,
                emission_trajectory[:, :, k],
            )

        return self.estimated_state

    def reset(self) -> None:
        self._init_batch_size(
            batch_size=self._batch_size, is_initialized=False
        )
