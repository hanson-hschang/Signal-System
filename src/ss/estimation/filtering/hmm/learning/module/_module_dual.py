from typing import Generic

import torch

from ss.estimation.filtering.hmm.learning.module.config import (
    LearningDualHmmFilterConfig,
)
from ss.estimation.filtering.hmm.learning.module.emission import EmissionModule
from ss.estimation.filtering.hmm.learning.module.estimation import (
    EstimationModule,
)
from ss.estimation.filtering.hmm.learning.module.filter import DualFilterModule
from ss.estimation.filtering.hmm.learning.module.transition import (
    DualTransitionModule,
)
from ss.utility.learning.module import BaseLearningModule
from ss.utility.learning.parameter.transformer import T
from ss.utility.learning.parameter.transformer.config import TC
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class LearningDualHmmFilter(
    BaseLearningModule[LearningDualHmmFilterConfig[TC]],
    Generic[T, TC],
):
    """
    `LearningHmmFilter` module for learning the hidden Markov model
    and estimating the next observation.
    """

    def __init__(
        self,
        config: LearningDualHmmFilterConfig[TC],
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
        self._filter = DualFilterModule(self._config.filter)

        # Define the learnable emission, transition and estimation modules
        self._emission = EmissionModule[T, TC](
            self._config.emission, self._config.filter
        )
        self._transition = DualTransitionModule[T, TC](
            self._config.transition, self._config.filter
        )
        self._estimation = EstimationModule[T, TC](
            self._config.estimation, self._config.filter
        )

        if self._estimation_matrix_binding:
            self._estimation.matrix_parameter.bind_with(
                self._emission.matrix_parameter
            )

        self.reset()

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
    def history_horizon(self) -> int:
        return self._filter.history_horizon

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
    def transition(self) -> DualTransitionModule[T, TC]:
        return self._transition

    @property
    def estimation(self) -> EstimationModule[T, TC]:
        return self._estimation

    @property
    def filter(self) -> DualFilterModule:
        return self._filter

    def forward(self, observation_trajectory: torch.Tensor) -> torch.Tensor:
        """
        forward method for the `LearningDualHmmFilter` class

        Parameters
        ----------
        observation_trajectory : torch.Tensor
            shape (batch_size, observation_dim=1, horizon,)

        Returns
        -------
        estimation_trajectory : torch.Tensor
            shape (batch_size, estimation_dim, horizon,)
        """

        emission_trajectory = self._emission(
            observation_trajectory,  # (batch_size, observation_dim=1, horizon)
        )  # (batch_size, state_dim, horizon)
        estimated_state_trajectory = self._transition(
            emission_trajectory
        )  # (batch_size, state_dim, horizon)
        estimation_trajectory: torch.Tensor = self._estimation(
            estimated_state_trajectory,
        )  # (batch_size, estimation_dim, horizon)

        return estimation_trajectory

    def reset(self, batch_size: int = 1) -> None:
        # self._is_initialized = False
        self.batch_size = batch_size
        self._emission.reset()
        self._transition.reset(batch_size=batch_size)
        self._estimation.reset()
        self._filter.reset(
            self._transition.initial_state, batch_size=batch_size
        )

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

        for k in range(emission_trajectory.shape[-1]):
            emission = emission_trajectory[
                :, :, k
            ]  # (batch_size, observation_dim=1)
            self._filter.update_emission(emission)

            estimated_state_distribution = self._transition.at_inference(
                self._filter.emission_difference_trajectory,
                self._filter.estimated_state_trajectory,
            )  # (batch_size, state_dim)

            # estimated_state_trajectory = self._forward(
            #     observation,
            # self._filter.get_emission_history(),
            # self._filter._emission_difference_history,
            # )
            # print(estimated_state_trajectory.shape)
            # if torch.isnan(estimated_state_trajectory).any():
            #     print(estimated_state_trajectory)
            #     quit()

            # logger.info(self.transition._control_trajectory_over_layers)

            self._filter.update_state(estimated_state_distribution)

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
            self._filter._estimated_state,  # (batch_size, state_dim)
        )
        return estimation.detach()
