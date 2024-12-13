from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LearningHiddenMarkovModelFilterParameters:
    state_dim: int
    observation_dim: int
    feature_dim: int = 1
    layer_dim: int = 1
    horizon_of_observation_history: int = 1


class LearningHiddenMarkovModelFilterBlock(nn.Module):
    def __init__(
        self,
        feature_id: int,
        params: LearningHiddenMarkovModelFilterParameters,
    ) -> None:
        super().__init__()
        self._feature_id = feature_id
        self._params = params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


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


class LearningHiddenMarkovModelFilter(nn.Module):
    def __init__(
        self,
        params: LearningHiddenMarkovModelFilterParameters,
    ) -> None:
        super().__init__()
        self._params = params
        self._observation_history = torch.zeros(
            (
                self._params.observation_dim,
                self._params.horizon_of_observation_history,
            ),
            dtype=torch.float64,
        )
        self.emission_layer = nn.Linear(
            self._params.state_dim, self._params.observation_dim
        )
        self.layers = nn.ModuleList()
        for layer_id in range(self._params.layer_dim):
            self.layers.append(
                LearningHiddenMarkovModelFilterLayer(layer_id, self._params)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    @torch.no_grad()
    def update(self, observation: torch.Tensor) -> None:
        self._observation_history = torch.roll(
            self._observation_history, 1, dims=1
        )
        self._observation_history[:, 0] = observation

    @torch.inference_mode()
    def estimate(
        self,
    ) -> torch.Tensor:
        x: torch.Tensor = self(self._observation_history)[:, 0]
        return x
