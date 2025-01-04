from typing import Any, Tuple

from dataclasses import dataclass

import torch
from torch import nn

from ss.learning import BaseLearningModule, BaseLearningParameters
from ss.utility.descriptor import TensorReadOnlyDescriptor
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


@dataclass
class LearningHiddenMarkovModelFilterParameters(BaseLearningParameters):
    state_dim: int
    observation_dim: int
    feature_dim: int = 1
    layer_dim: int = 1


class LearningHiddenMarkovModelFilterBlock(nn.Module):
    def __init__(
        self,
        feature_id: int,
        params: LearningHiddenMarkovModelFilterParameters,
    ) -> None:
        super().__init__()
        self._feature_id = feature_id
        self._params = params
        self._weight = nn.Parameter(
            torch.randn(
                (self._params.state_dim, self._params.state_dim),
                dtype=torch.float64,
            )
        )
        self._initial_state = nn.Parameter(
            torch.randn(self._params.state_dim, dtype=torch.float64)
        )

        self._is_initialized = False
        self._estimated_next_state = (
            torch.ones(self._params.state_dim, dtype=torch.float64)
            / self._params.state_dim
        )

        # self._cosine_transform_matrix = nn.Parameter(
        #     self._compute_cosine_transform_matrix(self._params.state_dim),
        #     requires_grad=False,
        # )

    @property
    def estimated_next_state(self) -> torch.Tensor:
        if self.training:
            self._is_initialized = False
            _estimated_next_state = nn.functional.softmax(
                self._initial_state, dim=0
            )
            return _estimated_next_state
        if not self._is_initialized:
            self._is_initialized = True
            self._estimated_next_state = nn.functional.softmax(
                self._initial_state, dim=0
            )
        return self._estimated_next_state

    @staticmethod
    def _compute_cosine_transform_matrix(dim: int) -> torch.Tensor:
        with torch.no_grad():
            # Create coordinate tensors
            i_coords = torch.arange(dim).float() + 0.5
            j_coords = torch.arange(dim).float() + 0.5

            # Compute outer product using einsum
            i_term = i_coords.view(-1, 1)  # Shape: (dim, 1)
            j_term = j_coords.view(1, -1)  # Shape: (1, dim)

            # Calculate cosine weights
            weight = torch.cos(torch.pi / dim * i_term * j_term)

            # Apply scaling factor
            weight = weight * torch.sqrt(torch.tensor(2.0 / dim))
        return torch.tensor(weight.detach().numpy(), dtype=torch.float64)

    def forward(self, emission_trajectory: torch.Tensor) -> torch.Tensor:

        batch_size, horizon_of_observation_trajectory, _ = (
            emission_trajectory.shape
        )
        # (batch_size, horizon_of_observation_trajectory, state_dim)
        estimated_next_state_trajectory = torch.zeros(
            (
                batch_size,
                horizon_of_observation_trajectory,
                self._params.state_dim,
            ),
            dtype=torch.float64,
        )

        transition_matrix = nn.functional.softmax(self._weight, dim=1)
        estimated_next_state = self.estimated_next_state.repeat(
            batch_size, 1
        )  # (batch_size, state_dim)

        for k in range(horizon_of_observation_trajectory):
            estimated_state = nn.functional.normalize(
                estimated_next_state * emission_trajectory[:, k, :],
                p=1,
                dim=1,
            )  # (batch_size, state_dim)

            estimated_next_state = torch.matmul(
                estimated_state,
                transition_matrix,
            )

            estimated_next_state_trajectory[:, k, :] = estimated_next_state

        if (not self.training) and (batch_size == 1):
            self._estimated_next_state = estimated_next_state.unsqueeze(0)

        return estimated_next_state_trajectory


class LearningHiddenMarkovModelFilterLayer(nn.Module):
    def __init__(
        self,
        layer_id: int,
        params: LearningHiddenMarkovModelFilterParameters,
    ) -> None:
        super().__init__()
        self._layer_id = layer_id
        self._params = params

        self.blocks = nn.ModuleList()
        for feature_id in range(self._params.feature_dim):
            self.blocks.append(
                LearningHiddenMarkovModelFilterBlock(feature_id, self._params)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sum = torch.zeros_like(x)
        for block in self.blocks:
            sum += block(x)
        sum /= self._params.feature_dim
        return sum


class LearningTransitionProcess(nn.Module):
    def __init__(
        self,
        params: LearningHiddenMarkovModelFilterParameters,
    ) -> None:
        super().__init__()
        self._params = params
        self.layers = nn.ModuleList()
        for layer_id in range(self._params.layer_dim):
            self.layers.append(
                LearningHiddenMarkovModelFilterLayer(layer_id, self._params)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class LearningHiddenMarkovModelFilter(BaseLearningModule):
    def __init__(
        self,
        params: LearningHiddenMarkovModelFilterParameters,
    ) -> None:
        super().__init__(params)

        # Define the dimensions of the state and observation
        self._state_dim = params.state_dim
        self._observation_dim = params.observation_dim

        # Define learnable parameters including the emission layer and transition layer
        self._emission_layer = nn.Linear(
            self._observation_dim,
            self._state_dim,
            bias=False,
            dtype=torch.float64,
        )
        self._transition_layer = LearningTransitionProcess(params)

        # Initialize the estimated next state, and next observation
        with torch.no_grad():
            self._estimated_next_state = (
                torch.ones(self._state_dim, dtype=torch.float64)
                / self._state_dim
            )
            self._estimated_next_observation = torch.matmul(
                self._estimated_next_state,
                self.emission_matrix,
            )

    estimated_next_state = TensorReadOnlyDescriptor("_state_dim")
    estimated_next_observation = TensorReadOnlyDescriptor("_observation_dim")

    @property
    def emission_matrix(self) -> torch.Tensor:
        _emission_matrix = nn.functional.softmax(
            self._emission_layer.weight, dim=1
        )
        return _emission_matrix

    def forward(self, observation_trajectory: torch.Tensor) -> torch.Tensor:
        """
        forward method for the LearningHiddenMarkovModelFilter class

        Parameters
        ----------
        observation_trajectory : torch.Tensor
            shape (batch_size, observation_dim, horizon_of_observation_trajectory)

        Returns
        -------
        estimated_next_observation_trajectory: torch.Tensor
            shape (batch_size, observation_dim, horizon_of_observation_trajectory)
        """
        estimated_next_observation_trajectory, _ = self._forward(
            observation_trajectory
        )
        return estimated_next_observation_trajectory

    def _forward(
        self,
        observation_trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        _forward method for the LearningHiddenMarkovModelFilter class

        Parameters
        ----------
        observation_trajectory : torch.Tensor
            shape (batch_size, horizon_of_observation_trajectory, observation_dim)

        Returns
        -------
        estimated_next_observation_trajectory: torch.Tensor
            shape (batch_size, horizon_of_observation_trajectory, observation_dim)
        estimated_next_state: torch.Tensor
            shape (batch_size, state_dim)
        """

        # Get emission_matrix
        emission_matrix = self.emission_matrix  # (state_dim, observation_dim)

        # Get indices of 1s in one-hot encoding
        observation_value_trajectory = torch.argmax(
            observation_trajectory,  # (batch_size, horizon_of_observation_trajectory, observation_dim)
            dim=2,
        )  # (batch_size, horizon_of_observation_trajectory)

        # Get emission probabilities for each observation in the trajectory
        emission_trajectory = torch.moveaxis(
            emission_matrix[:, observation_value_trajectory], 0, 2
        )  # (batch_size, horizon_of_observation_trajectory, state_dim)

        estimated_next_state_trajectory = self._transition_layer(
            emission_trajectory
        )  # (batch_size, horizon_of_observation_trajectory, state_dim)

        estimated_next_observation_trajectory = torch.matmul(
            estimated_next_state_trajectory,  # (batch_size, horizon_of_observation_trajectory, state_dim)
            emission_matrix,  # (state_dim, observation_dim)
        )  # (batch_size, horizon_of_observation_trajectory, observation_dim)

        estimated_next_state = estimated_next_state_trajectory[
            :, -1, :
        ]  # (batch_size, state_dim)

        return estimated_next_observation_trajectory, estimated_next_state

    @torch.no_grad()
    def update(self, observation_trajectory: torch.Tensor) -> None:
        if observation_trajectory.ndim == 1:
            observation_trajectory = observation_trajectory.unsqueeze(0)
        assert (observation_trajectory.ndim == 2) and (
            observation_trajectory.shape[1] == self._observation_dim
        ), (
            f"observation_trajectory must be in the shape of (horizon_of_observation_trajectory, observation_dim). "
            f"observation_trajectory given has the shape of {observation_trajectory.shape}."
        )

        _, estimated_next_state = self._forward(
            observation_trajectory.unsqueeze(0),
        )
        self._estimated_next_state = estimated_next_state.squeeze(0)

    @torch.inference_mode()
    def estimate(
        self,
    ) -> None:
        # Compute the estimated next observation
        self._estimated_next_observation = torch.matmul(
            self._estimated_next_state,
            self.emission_matrix,
        )
