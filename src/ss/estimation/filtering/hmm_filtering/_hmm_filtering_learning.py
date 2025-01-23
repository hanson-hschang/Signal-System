from typing import Any, Tuple, cast

import torch
from numpy.typing import ArrayLike
from torch import nn

from ss.estimation.filtering.hmm_filtering._hmm_filtering_data import (
    HiddenMarkovModelObservationDataset,
)
from ss.estimation.filtering.hmm_filtering._hmm_filtering_learning_config import (
    LearningHiddenMarkovModelFilterConfig,
)
from ss.estimation.filtering.hmm_filtering._hmm_filtering_learning_emission import (
    LearningHiddenMarkovModelFilterEmissionProcess,
)
from ss.estimation.filtering.hmm_filtering._hmm_filtering_learning_transition import (
    LearningHiddenMarkovModelFilterTransitionProcess,
)
from ss.learning import BaseLearningModule, BaseLearningProcess
from ss.utility.descriptor import BatchTensorReadOnlyDescriptor
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class LearningHiddenMarkovModelFilter(
    BaseLearningModule[LearningHiddenMarkovModelFilterConfig]
):
    """
    `LearningHiddenMarkovModelFilter` class for learning the hidden Markov model and estimating the next observation.
    """

    def __init__(
        self,
        config: LearningHiddenMarkovModelFilterConfig,
    ) -> None:
        """
        Initialize the `LearningHiddenMarkovModelFilter` class

        Arguments:
        ----------
            config : LearningHiddenMarkovModelFilterParameters
                dataclass containing the configuration for the `LearningHiddenMarkovModelFilter` class
        """
        super().__init__(config)

        # Define the dimensions of the state and estimation
        self._state_dim = self._config.state_dim
        self._discrete_observation_dim = self._config.discrete_observation_dim

        # Define learnable parameters including the emission layer and transition layer
        self._emission_process = (
            LearningHiddenMarkovModelFilterEmissionProcess(self._config)
        )
        # self._emission_process = nn.Linear(
        #     self._discrete_observation_dim,
        #     self._state_dim,
        #     bias=False,
        #     dtype=torch.float64,
        # )
        self._transition_process = (
            LearningHiddenMarkovModelFilterTransitionProcess(self._config)
        )

        # Initialize the estimated next state, and next observation for the inference mode
        self._init_batch_size(batch_size=1)

    def _init_batch_size(
        self, batch_size: int, is_initialized: bool = False
    ) -> None:
        self._is_initialized = is_initialized
        self._batch_size = batch_size
        with torch.no_grad():
            self._estimated_next_state_probability = (
                torch.ones(
                    (self._batch_size, self._state_dim), dtype=torch.float64
                )
                / self._state_dim
            )  # (batch_size, state_dim)
            self._estimated_next_observation_probability = torch.matmul(
                self._estimated_next_state_probability,
                self.emission_matrix,
            )  # (batch_size, observation_dim)

    estimated_next_state_probability = BatchTensorReadOnlyDescriptor(
        "_batch_size", "_state_dim"
    )
    estimated_next_observation_probability = BatchTensorReadOnlyDescriptor(
        "_batch_size", "_observation_dim"
    )

    @property
    def transition_process(
        self,
    ) -> LearningHiddenMarkovModelFilterTransitionProcess:
        return self._transition_process

    @property
    def emission_process(
        self,
    ) -> LearningHiddenMarkovModelFilterEmissionProcess:
        return self._emission_process

    @property
    def emission_matrix(self) -> torch.Tensor:
        _emission_process_layer = cast(
            nn.Linear, self._emission_process.layers[0]
        )
        _emission_matrix = nn.functional.softmax(
            _emission_process_layer.weight, dim=1
        )
        return _emission_matrix

    def forward(self, observation_trajectory: torch.Tensor) -> torch.Tensor:
        """
        forward method for the `LearningHiddenMarkovModelFilter` class

        Arguments:
        ----------
            observation_trajectory : torch.Tensor
                shape (batch_size, horizon)

        Returns:
        --------
            estimated_next_observation_log_probability_trajectory : torch.Tensor
                shape (batch_size, discrete_observation_dim, horizon)
        """
        estimated_next_observation_probability_trajectory, _ = self._forward(
            observation_trajectory
        )  # (batch_size, horizon, discrete_observation_dim)
        estimated_next_observation_log_probability_trajectory = torch.moveaxis(
            torch.log(estimated_next_observation_probability_trajectory), 1, 2
        )  # (batch_size, discrete_observation_dim, horizon)
        return estimated_next_observation_log_probability_trajectory

    def _forward(
        self,
        observation_trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        _forward method for the `LearningHiddenMarkovModelFilter` class

        Arguments:
        ----------
            observation_trajectory : torch.Tensor
                shape (batch_size, horizon)

        Returns:
        --------
            estimated_next_observation_probability_trajectory: torch.Tensor
                shape (batch_size, horizon, observation_dim)
            estimated_next_state_probability: torch.Tensor
                shape (batch_size, state_dim)
        """

        # Get emission_matrix
        emission_matrix = (
            self.emission_matrix
        )  # (state_dim, discrete_observation_dim)

        # Get emission probabilities for each observation in the trajectory
        embedding_state_trajectory = torch.moveaxis(
            emission_matrix[:, observation_trajectory], 0, 2
        )  # (batch_size, horizon, state_dim)

        estimated_next_state_probability_trajectory = self._transition_process(
            embedding_state_trajectory
        )  # (batch_size, horizon, state_dim)

        estimated_next_observation_probability_trajectory = torch.matmul(
            estimated_next_state_probability_trajectory,  # (batch_size, horizon, state_dim)
            emission_matrix,  # (state_dim, observation_dim)
        )  # (batch_size, horizon, observation_dim)

        estimated_next_state_probability = (
            estimated_next_state_probability_trajectory[:, -1, :]
        )  # (batch_size, state_dim)

        return (
            estimated_next_observation_probability_trajectory,
            estimated_next_state_probability,
        )

    def reset(self) -> None:
        self._is_initialized = False
        self._emission_process.reset()
        self._transition_process.reset()

    def _check_batch_size(self, batch_size: int) -> None:
        if self._is_initialized:
            assert batch_size == self._batch_size, (
                f"batch_size must be the same as the initialized batch_size. "
                f"batch_size given is {batch_size} while the initialized batch_size is {self._batch_size}."
            )
            return
        self._init_batch_size(batch_size, is_initialized=True)

    @torch.inference_mode()
    def update(self, observation_trajectory: torch.Tensor) -> None:
        """
        Update the estimated next state probability based on the observation trajectory.
        Use the `estimated_next_state_probability` property to get the probability of the estimated next state.

        Arguments:
        ----------
            observation_trajectory : torch.Tensor
                shape = (horizon,) or (batch_size, horizon)
        """
        if observation_trajectory.ndim == 0:
            observation_trajectory = observation_trajectory.unsqueeze(
                0
            )  # (horizon=1,)
        if observation_trajectory.ndim == 1:
            observation_trajectory = observation_trajectory.unsqueeze(
                0
            )  # (batch_size=1, horizon)
        assert observation_trajectory.ndim == 2, (
            f"observation_trajectory must be in the shape of (batch_size, horizon). "
            f"observation_trajectory given has the shape of {observation_trajectory.shape}."
        )
        self._check_batch_size(batch_size=observation_trajectory.shape[0])

        _, estimated_next_state_probability = self._forward(
            observation_trajectory
        )
        self._estimated_next_state_probability = (
            estimated_next_state_probability  # (batch_size, state_dim)
        )

    @torch.inference_mode()
    def estimate(self) -> None:
        """
        Compute the probability of the estimated next observation. This method should be called after the `update` method.
        Use the `estimated_next_observation_probability` property to get the probability of the estimated next observation.
        """

        self._estimated_next_observation_probability = torch.matmul(
            self._estimated_next_state_probability,
            self.emission_matrix,
        )

    @torch.inference_mode()
    def predict(self, horizon: int) -> torch.Tensor:
        """
        Predict the next observation(s) for the given horizon.

        Arguments
        ---------
        horizon : int
            The horizon to predict the next observation(s).

        Returns
        -------
        predicted_observation: torch.Tensor
            shape = (batch_size, horizon)
        """
        return torch.zeros((self._batch_size, horizon), dtype=torch.float64)


class LearningHiddenMarkovModelFilterProcess(BaseLearningProcess):
    def _evaluate_one_batch(self, data_batch: Any) -> torch.Tensor:
        (
            observation_trajectory,
            next_observation_trajectory,
        ) = HiddenMarkovModelObservationDataset.from_batch(
            data_batch
        )  # (batch_size, max_length), (batch_size, max_length)
        estimated_next_observation_probability_trajectory = self._model(
            observation_trajectory=observation_trajectory
        )  # (batch_size, discrete_observation_dim, max_length)
        _loss: torch.Tensor = self._loss_function(
            estimated_next_observation_probability_trajectory,
            next_observation_trajectory,  # (batch_size, max_length)
        )
        return _loss
