from typing import Callable, Generic, Tuple, assert_never

import torch
import torch.nn as nn

from ss.estimation.dual_filtering.hmm.learning.module.emission.config import (
    DualEmissionConfig,
)
from ss.estimation.dual_filtering.hmm.learning.module.filter.config import (
    DualFilterConfig,
)
from ss.utility.learning.module import BaseLearningModule
from ss.utility.learning.parameter.probability import ProbabilityParameter
from ss.utility.learning.parameter.transformer import T
from ss.utility.learning.parameter.transformer.config import TC
from ss.utility.map import transform


@torch.compile
def to_probability(
    estimation_trajectory: torch.Tensor,
) -> torch.Tensor:
    return nn.functional.normalize(
        estimation_trajectory,
        p=1,
        dim=-1,
    )


class DualEmissionModule(
    BaseLearningModule[DualEmissionConfig[TC]], Generic[T, TC]
):
    def __init__(
        self,
        config: DualEmissionConfig[TC],
        filter_config: DualFilterConfig,
    ) -> None:
        super().__init__(config)
        self._state_dim = filter_config.state_dim
        self._discrete_observation_dim = filter_config.discrete_observation_dim

        self._matrix = ProbabilityParameter[T, TC](
            self._config.matrix.probability_parameter,
            (self._state_dim, self._discrete_observation_dim),
        )

        self._forward: Callable[[torch.Tensor], torch.Tensor]
        self._init_forward()

    def _init_forward(self) -> None:
        match self._config.observation.option:
            case self._config.observation.Option.CATEGORY:
                self._forward = self._forward_category
            case self._config.observation.Option.PROBABILITY:
                self._forward = self._forward_probability
            case _ as _option:
                assert_never(_option)

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

    def _forward_category(
        self,
        observation_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        # observation_trajectory: (batch_size, horizon)

        emission_matrix = self.matrix  # (state_dim, discrete_observation_dim)

        # Get emission based on each observation in the trajectory
        emission_trajectory = torch.moveaxis(
            emission_matrix[:, observation_trajectory], 0, 2
        )  # (batch_size, horizon, state_dim)

        return emission_trajectory

    def _forward_probability(
        self,
        observation_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        # observation_trajectory: (batch_size, horizon, discrete_observation_dim)

        emission_matrix = self.matrix  # (state_dim, discrete_observation_dim)

        emission_trajectory = torch.matmul(
            observation_trajectory,  # (batch_size, horizon, discrete_observation_dim)
            emission_matrix.T,  # (discrete_observation_dim, state_dim)
        )  # (batch_size, horizon, state_dim)

        return emission_trajectory

    def forward(
        self,
        observation_trajectory: torch.Tensor,
    ) -> torch.Tensor:

        # shape of observation_trajectory is based on the observation option
        # if the option is CATEGORY, then the shape is (batch_size, horizon)
        # if the option is PROBABILITY, then the shape is (batch_size, horizon, discrete_observation_dim)

        # emission_trajectory = nn.functional.normalize(
        #     self._forward(observation_trajectory),
        #     p=1,
        #     dim=-1,
        # )  # (batch_size, horizon, state_dim)

        emission_trajectory = self._forward(observation_trajectory)
        # (batch_size, horizon, state_dim)

        # emission_difference_trajectory = (
        #     emission_trajectory * 2 - 1
        # )  # (batch_size, horizon, state_dim)

        # likelihood_trajectory = transform(
        #     emission_trajectory,
        #     to_probability,
        # )  # (batch_size, horizon, state_dim)

        return emission_trajectory

    def validate_observation_shape(
        self, observation: torch.Tensor, batch_size: int = 1
    ) -> torch.Tensor:
        match self._config.observation.option:
            case self._config.observation.Option.CATEGORY:
                if observation.ndim == 0:
                    observation = observation.unsqueeze(0)  # (horizon=1,)
                if observation.ndim == 1:
                    if batch_size == 1:
                        observation = observation.unsqueeze(
                            0
                        )  # (batch_size=1, horizon)
                    else:
                        observation = observation.unsqueeze(
                            1
                        )  # (batch_size, horizon=1)
                assert observation.ndim == 2, (
                    f"observation must be in the shape of (batch_size, horizon). "
                    f"observation given has the shape of {observation.shape}."
                )
            case self._config.observation.Option.PROBABILITY:
                if observation.ndim == 1:
                    observation = observation.unsqueeze(
                        0
                    )  # (horizon=1, discrete_observation_dim)
                if observation.ndim == 2:
                    if batch_size == 1:
                        observation = observation.unsqueeze(
                            0
                        )  # (batch_size=1, horizon, discrete_observation_dim)
                    else:
                        observation = observation.unsqueeze(
                            1
                        )  # (batch_size, horizon=1, discrete_observation_dim)
                assert observation.ndim == 3, (
                    f"observation must be in the shape of (batch_size, horizon, discrete_observation_dim). "
                    f"observation given has the shape of {observation.shape}."
                )
            case _ as _option:
                assert_never(_option)
        return observation
