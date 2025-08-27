from typing import Generic, Tuple, assert_never

import torch

from ss.estimation.dual_filtering.hmm.learning.module.config import (
    LearningDualHmmFilterConfig,
)
from ss.estimation.dual_filtering.hmm.learning.module.emission import (
    DualEmissionModule,
)
from ss.estimation.dual_filtering.hmm.learning.module.estimation import (
    DualEstimationModule,
)
from ss.estimation.dual_filtering.hmm.learning.module.filter import (
    DualFilterModule,
)
from ss.estimation.dual_filtering.hmm.learning.module.transition import (
    DualTransitionModule,
)
from ss.utility.descriptor import (
    BatchTensorReadOnlyDescriptor,
    Descriptor,
    ReadOnlyDescriptor,
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
    `LearningHmmFilter` module for learning the hidden Markov model and estimating the next observation.
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
            dataclass containing the configuration for the module `LearningHmmFilter` class
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
        self._emission = DualEmissionModule[T, TC](
            self._config.emission, self._config.filter
        )
        self._transition = DualTransitionModule[T, TC](
            self._config.transition, self._config.filter
        )
        self._estimation = DualEstimationModule[T, TC](
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
    def history_length(self) -> int:
        return self._filter.history_length

    @property
    def batch_size(self) -> int:
        return self._filter.batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        self._filter.batch_size = batch_size

    @property
    def emission(self) -> DualEmissionModule[T, TC]:
        return self._emission

    @property
    def transition(self) -> DualTransitionModule[T, TC]:
        return self._transition

    @property
    def estimation(self) -> DualEstimationModule[T, TC]:
        return self._estimation

    def forward(self, observation_trajectory: torch.Tensor) -> torch.Tensor:
        """
        forward method for the `LearningHmmFilter` class

        Parameters
        ----------
        observation_trajectory : torch.Tensor
            shape (batch_size, horizon)

        Returns
        -------
        estimation_trajectory : torch.Tensor
            shape (batch_size, horizon, estimation_dim)
        """

        emission_trajectory = self._emission(observation_trajectory)

        estimated_state_trajectory = self._forward(
            emission_trajectory,  # (batch_size, horizon, state_dim)
            # emission_difference_trajectory,  # (batch_size, horizon, state_dim)
            # observation_trajectory  # (batch_size, horizon)
        )

        estimation_trajectory: torch.Tensor = self._estimation(
            estimated_state_trajectory,  # (batch_size, horizon, state_dim)
            # predicted_state_trajectory,  # (batch_size, horizon, state_dim)
        )  # (batch_size, horizon, estimation_dim)

        return estimation_trajectory

    def _forward(
        self,
        emission_trajectory: torch.Tensor,
        # emission_difference_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """
        _forward method for the `LearningHmmFilter` class

        Arguments
        ---------
        observation_trajectory : torch.Tensor
            shape (batch_size, horizon)

        Returns
        -------
        estimated_state_trajectory: torch.Tensor
            shape (batch_size, horizon, state_dim)
        predicted_state_trajectory: torch.Tensor
            shape (batch_size, horizon, state_dim)
        """

        # emission_trajectory, emission_difference_trajectory = self._emission(
        #     observation_trajectory
        # )
        # emission_difference_trajectory = self._filter.compute_emission_difference(
        #     emission_trajectory,  # (batch_size, horizon, state_dim)
        # )

        estimated_state_trajectory: torch.Tensor = self._transition(
            emission_trajectory
        )  # (batch_size, horizon, state_dim)

        # return estimated_state_trajectory, estimation_trajectory

        return estimated_state_trajectory

    def reset(self) -> None:
        # self._is_initialized = False
        self._emission.reset()
        self._transition.reset()
        self._estimation.reset()

    @torch.inference_mode()
    def update(self, observation: torch.Tensor) -> None:
        """
        Update the estimated state based on the observation.

        Arguments:
        ----------
        observation : torch.Tensor
            shape = (batch_size, horizon) or (batch_size, horizon, discrete_observation_dim)
        """
        observation = self._emission.validate_observation_shape(
            observation, batch_size=self.batch_size
        )

        emission = self._emission(observation)

        self._filter.update_history(emission)

        estimated_state_trajectory = self._forward(
            self._filter.get_emission_history(),
            # self._filter._emission_difference_history,
        )
        # print(estimated_state_trajectory.shape)
        # if torch.isnan(estimated_state_trajectory).any():
        #     print(estimated_state_trajectory)
        #     quit()

        # logger.info(self.transition._control_trajectory_over_layers)

        self._filter._estimated_state_history = estimated_state_trajectory
        self._filter.estimated_state = estimated_state_trajectory[:, -1, :]
        # self._filter.predicted_state = predicted_state_trajectory[:, -1, :]

    @torch.inference_mode()
    def estimate(self) -> torch.Tensor:
        """
        Compute the estimation. This method should be called after the `update` method.

        Returns
        -------
        estimation: torch.Tensor
            Based on the estimation option in the configuration, the chosen estimation will be returned.
        """
        # self._estimation = self._estimate()

        # return self._estimation.result
        estimation: torch.Tensor = self._estimation(
            self._filter.estimated_state,  # (batch_size, state_dim)
            # self._filter.predicted_state,  # (batch_size, state_dim)
        )
        return estimation.detach()
