from typing import Optional, Tuple, assert_never

from enum import StrEnum

import torch
from torch import nn

from ss.learning import BaseLearningModule


class BaseLearningHmmFilterBlock(BaseLearningModule):
    def __init__(
        self,
        feature_id: int,
        state_dim: int,
    ) -> None:
        super().__init__()
        self._feature_id = feature_id
        self._state_dim = state_dim
        self._initial_state = nn.Parameter(
            torch.randn(self._state_dim, dtype=torch.float64)
        )  # (state_dim,)
        self._is_initialized = False
        self._estimated_previous_state = (
            torch.ones(self._state_dim, dtype=torch.float64) / self._state_dim
        ).repeat(
            1, 1
        )  # (batch_size, state_dim)

    def get_estimated_previous_state(
        self, batch_size: int = 1
    ) -> torch.Tensor:
        if not self.inference:
            self._is_initialized = False
            _estimated_previous_state = nn.functional.softmax(
                self._initial_state, dim=0
            ).repeat(batch_size, 1)
            return _estimated_previous_state
        if not self._is_initialized:
            self._is_initialized = True
            self._estimated_previous_state = nn.functional.softmax(
                self._initial_state, dim=0
            ).repeat(batch_size, 1)
        return self._estimated_previous_state

    def reset(self) -> None:
        self._is_initialized = False

    @property
    def transition_matrix(self) -> torch.Tensor:
        return torch.eye(self._state_dim, dtype=torch.float64)

    def _prediction_step(
        self,
        previous_estimated_state: torch.Tensor,
        transition_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if transition_matrix is None:
            transition_matrix = self.transition_matrix
        predicted_state = torch.matmul(
            previous_estimated_state, transition_matrix
        )
        return predicted_state

    def _update_step(
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

    def forward(
        self, likelihood_state_trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, horizon, _ = likelihood_state_trajectory.shape
        # (batch_size, horizon, state_dim)

        estimated_state_trajectory = torch.zeros(
            (batch_size, horizon, self._state_dim),
            dtype=torch.float64,
        )

        predicted_next_state_trajectory = torch.zeros(
            (batch_size, horizon, self._state_dim),
            dtype=torch.float64,
        )

        transition_matrix = self.transition_matrix  # (state_dim, state_dim)

        estimated_previous_state = self.get_estimated_previous_state(
            batch_size
        )  # (batch_size, state_dim)

        # prediction step based on model process (predicted probability)
        predicted_state = self._prediction_step(
            estimated_previous_state,
            transition_matrix,
        )  # (batch_size, state_dim)

        for k in range(horizon):

            # update step based on input_state (conditional probability)
            estimated_state = self._update_step(
                predicted_state,
                likelihood_state_trajectory[:, k, :],
            )  # (batch_size, state_dim)

            # prediction step based on model process (predicted probability)
            predicted_next_state = self._prediction_step(
                estimated_state, transition_matrix
            )  # (batch_size, state_dim)

            estimated_state_trajectory[:, k, :] = estimated_state
            predicted_next_state_trajectory[:, k, :] = predicted_next_state

            estimated_previous_state = estimated_state
            predicted_state = predicted_next_state

        if self.inference:
            self._estimated_previous_state = estimated_previous_state

        return estimated_state_trajectory, predicted_next_state_trajectory


class LearningHmmFilterFullMatrixBlock(BaseLearningHmmFilterBlock):
    def __init__(
        self,
        feature_id: int,
        state_dim: int,
    ) -> None:
        super().__init__(feature_id, state_dim)
        self._weight = nn.Parameter(
            torch.randn(
                (self._state_dim, self._state_dim),
                dtype=torch.float64,
            )
        )

    @property
    def transition_matrix(self) -> torch.Tensor:
        return nn.functional.softmax(self._weight, dim=1)


class LearningHmmFilterSpatialInvariantBlock(BaseLearningHmmFilterBlock):
    def __init__(
        self,
        feature_id: int,
        state_dim: int,
    ) -> None:
        super().__init__(feature_id, state_dim)
        self._weight = nn.Parameter(
            torch.randn(
                self._state_dim,
                dtype=torch.float64,
            )
        )

    @property
    def transition_matrix(self) -> torch.Tensor:
        matrix = torch.empty(
            (self._state_dim, self._state_dim), dtype=torch.float64
        )

        matrix[0, :] = nn.functional.softmax(self._weight, dim=0)
        for i in range(1, self._state_dim):
            matrix[i, :] = torch.roll(matrix[i - 1, :], shifts=1)
        return matrix


class LearningHmmFilterBlockOption(StrEnum):
    FULL_MATRIX = "FULL_MATRIX"
    SPATIAL_INVARIANT = "SPATIAL_INVARIANT"

    @classmethod
    def get_block(
        cls,
        feature_id: int,
        state_dim: int,
        block_option: "LearningHmmFilterBlockOption",
    ) -> BaseLearningHmmFilterBlock:
        match block_option:
            case cls.FULL_MATRIX:
                return LearningHmmFilterFullMatrixBlock(feature_id, state_dim)
            case cls.SPATIAL_INVARIANT:
                return LearningHmmFilterSpatialInvariantBlock(
                    feature_id, state_dim
                )
            case _ as _invalid_block_option:
                assert_never(_invalid_block_option)
