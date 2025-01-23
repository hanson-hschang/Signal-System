from typing import assert_never

from enum import StrEnum

import torch
from torch import nn

from ss.learning import BaseLearningModule


class BaseLearningMarkovModelFilterBlock(BaseLearningModule):
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
        )
        self._is_initialized = False
        self._estimated_next_state_probability = (
            torch.ones(self._state_dim, dtype=torch.float64) / self._state_dim
        )

    @property
    def estimated_next_state_probability(self) -> torch.Tensor:
        if self.training:
            self._is_initialized = False
            _estimated_next_state_probability = nn.functional.softmax(
                self._initial_state, dim=0
            )
            return _estimated_next_state_probability
        if not self._is_initialized:
            self._is_initialized = True
            self._estimated_next_state_probability = nn.functional.softmax(
                self._initial_state, dim=0
            )
        return self._estimated_next_state_probability

    def reset(self) -> None:
        self._is_initialized = False

    def get_transition_matrix(self) -> torch.Tensor:
        return torch.eye(self._state_dim, dtype=torch.float64)

    def forward(self, emission_trajectory: torch.Tensor) -> torch.Tensor:
        batch_size, horizon, _ = emission_trajectory.shape
        # (batch_size, horizon, state_dim)
        estimated_next_state_probability_trajectory = torch.zeros(
            (batch_size, horizon, self._state_dim),
            dtype=torch.float64,
        )

        transition_matrix = self.get_transition_matrix()
        estimated_next_state_probability = (
            self.estimated_next_state_probability.repeat(batch_size, 1)
        )  # (batch_size, state_dim)

        for k in range(horizon):
            unnormalized_conditional_probability = (
                estimated_next_state_probability * emission_trajectory[:, k, :]
            )
            estimated_state_probability = nn.functional.normalize(
                unnormalized_conditional_probability,
                p=1,
                dim=1,
            )  # (batch_size, state_dim)

            estimated_next_state_probability = torch.matmul(
                estimated_state_probability,
                transition_matrix,
            )  # (batch_size, state_dim)

            estimated_next_state_probability_trajectory[:, k, :] = (
                estimated_next_state_probability
            )

        if self.inference:
            self._estimated_next_state_probability = (
                estimated_next_state_probability.squeeze(0)
            )

        return estimated_next_state_probability_trajectory


class LearningHiddenMarkovModelFilterFullMatrixBlock(
    BaseLearningMarkovModelFilterBlock
):
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

    def get_transition_matrix(self) -> torch.Tensor:
        return nn.functional.softmax(self._weight, dim=1)


class LearningHiddenMarkovModelFilterSpatialInvariantBlock(
    BaseLearningMarkovModelFilterBlock
):
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

    def get_transition_matrix(self) -> torch.Tensor:
        matrix = torch.empty(
            (self._state_dim, self._state_dim), dtype=torch.float64
        )
        matrix[0, :] = nn.functional.softmax(self._weight, dim=0)
        for i in range(1, self._state_dim):
            matrix[i, :] = torch.roll(matrix[i - 1, :], shifts=1)
        return matrix


class LearningHiddenMarkovModelFilterBlockOption(StrEnum):
    FULL_MATRIX = "FULL_MATRIX"
    SPATIAL_INVARIANT = "SPATIAL_INVARIANT"

    @classmethod
    def get_block(
        cls,
        feature_id: int,
        state_dim: int,
        block_option: "LearningHiddenMarkovModelFilterBlockOption",
    ) -> BaseLearningMarkovModelFilterBlock:
        match block_option:
            case cls.FULL_MATRIX:
                return LearningHiddenMarkovModelFilterFullMatrixBlock(
                    feature_id, state_dim
                )
            case cls.SPATIAL_INVARIANT:
                return LearningHiddenMarkovModelFilterSpatialInvariantBlock(
                    feature_id, state_dim
                )
            case _ as _invalid_block_option:
                assert_never(_invalid_block_option)
