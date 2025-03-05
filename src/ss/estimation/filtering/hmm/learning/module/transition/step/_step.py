from typing import Any, Tuple

import torch
import torch.nn as nn

from ss.estimation.filtering.hmm.learning.module import config as Config
from ss.utility.learning.parameter.probability import ProbabilityParameter


class TransitionStepMixin(nn.Module):
    def __init__(
        self,
        initial_state_config: Config.TransitionInitialStateConfig,
        state_dim: int,
        skip_first_transition: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._state_dim = state_dim
        self._skip_first_transition = skip_first_transition

        self._initial_state = ProbabilityParameter(
            initial_state_config.probability_parameter,
            (state_dim,),
        )

        self._is_initialized = False
        self._estimated_state = (
            torch.ones(self._state_dim) / self._state_dim
        ).repeat(
            1, 1
        )  # (batch_size, state_dim)
        self._matrix: ProbabilityParameter

    @property
    def initial_state_parameter(self) -> ProbabilityParameter:
        return self._initial_state

    @property
    def initial_state(self) -> torch.Tensor:
        initial_state: torch.Tensor = self._initial_state()
        return initial_state

    @initial_state.setter
    def initial_state(self, initial_state: torch.Tensor) -> None:
        self._initial_state.set_value(initial_state)

    @property
    def matrix_parameter(self) -> ProbabilityParameter:
        return self._matrix

    @property
    def matrix(self) -> torch.Tensor:
        raise NotImplementedError

    @matrix.setter
    def matrix(self, matrix: torch.Tensor) -> None:
        raise NotImplementedError

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

    def reset(self) -> None:
        self._is_initialized = False
        self._get_internal_estimated_state()
        self._is_initialized = False

    def _prediction(
        self,
        estimated_state: torch.Tensor,
        transition_matrix: torch.Tensor,
    ) -> torch.Tensor:
        predicted_next_state = torch.matmul(estimated_state, transition_matrix)
        return predicted_next_state

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

    def _first_prediction(
        self,
        estimated_state: torch.Tensor,
        transition_matrix: torch.Tensor,
    ) -> torch.Tensor:
        predicted_next_state = (
            estimated_state
            if self._skip_first_transition
            else self._prediction(estimated_state, transition_matrix)
        )  # (batch_size, state_dim)
        return predicted_next_state

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
        self, estimated_state: torch.Tensor, predicted_next_state: torch.Tensor
    ) -> None:
        self._estimated_state = (
            predicted_next_state
            if self._skip_first_transition
            else estimated_state
        )

    def process(
        self, likelihood_state_trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, horizon, _ = likelihood_state_trajectory.shape
        # (batch_size, horizon, state_dim)

        estimated_state_trajectory = torch.zeros(
            (batch_size, horizon, self._state_dim),
            device=likelihood_state_trajectory.device,
        )

        predicted_next_state_trajectory = torch.zeros(
            (batch_size, horizon, self._state_dim),
            device=likelihood_state_trajectory.device,
        )

        transition_matrix = self.matrix  # (state_dim, state_dim)

        estimated_state = self._get_internal_estimated_state(
            batch_size
        )  # (batch_size, state_dim)

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

        predicted_next_state = self._first_prediction(
            estimated_state, transition_matrix
        )  # (batch_size, state_dim)

        for k in range(horizon):
            likelihood_state = likelihood_state_trajectory[
                :, k, :
            ]  # (batch_size, state_dim)

            # update step based on input_state (conditional probability)
            estimated_state = self._update(
                predicted_next_state, likelihood_state
            )  # (batch_size, state_dim)

            # prediction step based on model process (predicted probability)
            predicted_next_state = self._prediction(
                estimated_state, transition_matrix
            )  # (batch_size, state_dim)

            estimated_state_trajectory[:, k, :] = estimated_state
            predicted_next_state_trajectory[:, k, :] = predicted_next_state

        # if self._inference:
        #     self._estimated_state = (
        #         predicted_next_state
        #         if self._skip_first_transition
        #         else estimated_state
        #     )

        self._set_internal_estimated_state(
            estimated_state, predicted_next_state
        )

        return estimated_state_trajectory, predicted_next_state_trajectory
