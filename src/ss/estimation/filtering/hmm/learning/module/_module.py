from typing import Generic

import torch

from ss.estimation.filtering.hmm.learning.module.config import (
    LearningHmmFilterConfig,
)
from ss.estimation.filtering.hmm.learning.module.emission import EmissionModule
from ss.estimation.filtering.hmm.learning.module.estimation import (
    EstimationModule,
)
from ss.estimation.filtering.hmm.learning.module.filter import FilterModule
from ss.estimation.filtering.hmm.learning.module.transition import (
    TransitionModule,
)
from ss.utility.learning.module import BaseLearningModule
from ss.utility.learning.parameter.transformer import T
from ss.utility.learning.parameter.transformer.config import TC
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class LearningHmmFilter(
    BaseLearningModule[LearningHmmFilterConfig[TC]],
    Generic[T, TC],
):
    """
    `LearningHmmFilter` module for learning the hidden Markov model
    and estimating the next observation.
    """

    def __init__(
        self,
        config: LearningHmmFilterConfig[TC],
    ) -> None:
        """
        Initialize the `LearningHmmFilter` module.

        Arguments:
        ----------
        config : LearningHmmFilterConfig
            dataclass containing the configuration for the
            module `LearningHmmFilter` class
        """
        super().__init__(config)

        self._estimation_matrix_binding = False
        if self._config.filter.estimation_dim == 0:
            self._estimation_matrix_binding = True
            self._config.filter.estimation_dim = (
                self._config.filter.discrete_observation_dim
            )

        # Define the filter module
        self._filter = FilterModule(self._config.filter)

        # Define the learnable emission, transition and estimation modules
        self._emission = EmissionModule[T, TC](
            self._config.emission, self._config.filter
        )
        self._transition = TransitionModule[T, TC](
            self._config.transition, self._config.filter
        )
        self._estimation = EstimationModule[T, TC](
            self._config.estimation, self._config.filter
        )

        if self._estimation_matrix_binding:
            self._estimation.matrix_parameter.bind_with(
                self._emission.matrix_parameter
            )

    @property
    def state_dim(self) -> int:
        return self._filter.state_dim

    @property
    def discrete_observation_dim(self) -> int:
        return self._filter.discrete_observation_dim

    @property
    def estimation_dim(self) -> int:
        return self._filter.estimation_dim

    @property
    def batch_size(self) -> int:
        return self._filter.batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        self._filter.batch_size = batch_size

    @property
    def emission(self) -> EmissionModule[T, TC]:
        return self._emission

    @property
    def transition(self) -> TransitionModule[T, TC]:
        return self._transition

    @property
    def estimation(self) -> EstimationModule[T, TC]:
        return self._estimation

    @property
    def filter(self) -> FilterModule:
        return self._filter

    def reset(self) -> None:
        self._emission.reset()
        self._transition.reset()
        self._estimation.reset()

    def forward(self, observation_trajectory: torch.Tensor) -> torch.Tensor:
        """
        forward method for the `LearningHmmFilter` class

        Parameters
        ----------
        observation_trajectory : torch.Tensor
            shape (batch_size, observation_dim=1, horizon,)

        Returns
        -------
        estimation_trajectory : torch.Tensor
            shape (batch_size, estimation_dim, horizon,)
        """

        emission_trajectory: torch.Tensor = self._emission(
            observation_trajectory,  # (batch_size, observation_dim=1, horizon)
        )  # (batch_size, state_dim, horizon,)
        # if self.training:
        #     noise = (
        #         torch.randn_like(
        #             emission_trajectory, device=emission_trajectory.device
        #         )
        #         * 1e-6
        #     )
        #     emission_trajectory = torch.clip(
        #         emission_trajectory + noise, min=0.0, max=1.0
        #     )
        estimated_state_trajectory = self._transition(
            emission_trajectory
        )  # (batch_size, state_dim, horizon,)
        estimation_trajectory: torch.Tensor = self._estimation(
            estimated_state_trajectory,
        )  # (batch_size, estimation_dim, horizon,)

        return estimation_trajectory

    @torch.inference_mode()
    def update(self, observation: torch.Tensor) -> None:
        """
        Update the estimated state based on the observation.

        Arguments:
        ----------
        observation : torch.Tensor
            shape = (batch_size, observation_dim=1, horizon) or
            (batch_size, observation_dim=1,) or (observation_dim=1,)
        """
        emission_trajectory = self._emission.at_inference(
            observation,  # (batch_size, observation_dim=1, horizon,)
            batch_size=self.batch_size,
        )  # (batch_size, state_dim, horizon,)
        self._filter.estimated_state = self._transition.at_inference(
            emission_trajectory
        )  # (batch_size, state_dim, horizon,)

    @torch.inference_mode()
    def estimate(self) -> torch.Tensor:
        """
        Compute the estimation.
        This method should be called after the `update` method.

        Returns
        -------
        estimation: torch.Tensor
            Based on the estimation option in the configuration,
            the chosen estimation will be returned.
        """
        estimation: torch.Tensor = self._estimation.at_inference(
            self._filter.estimated_state,  # (batch_size, state_dim)
        )
        return estimation.detach()
