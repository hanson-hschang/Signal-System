from typing import Any, Optional, Sequence, Tuple, assert_never

import torch
from numpy.typing import ArrayLike
from torch import nn

from ss.estimation.filtering.hmm_filtering._hmm_filtering_data import (
    HmmObservationDataset,
)
from ss.estimation.filtering.hmm_filtering._hmm_filtering_learning_config import (
    LearningHmmFilterConfig,
    LearningHmmFilterEstimationOption,
)
from ss.estimation.filtering.hmm_filtering._hmm_filtering_learning_emission import (
    LearningHmmFilterEmissionProcess,
)
from ss.estimation.filtering.hmm_filtering._hmm_filtering_learning_transition import (
    LearningHmmFilterTransitionProcess,
)
from ss.learning import BaseLearningModule, BaseLearningProcess
from ss.utility.descriptor import (
    BatchTensorReadOnlyDescriptor,
    ReadOnlyDescriptor,
)
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class LearningHmmFilter(BaseLearningModule[LearningHmmFilterConfig]):
    """
    `LearningHmmFilter` class for learning the hidden Markov model and estimating the next observation.
    """

    def __init__(
        self,
        config: LearningHmmFilterConfig,
    ) -> None:
        """
        Initialize the `LearningHmmFilter` class

        Arguments:
        ----------
            config : LearningHmmFilterParameters
                dataclass containing the configuration for the `LearningHmmFilter` class
        """
        super().__init__(config)

        # Define the dimensions of the state, observation, and the number of layers
        self._state_dim = self._config.state_dim
        self._discrete_observation_dim = self._config.discrete_observation_dim
        self._layer_dim = self._config.layer_dim

        # Define the learnable emission process and transition process
        self._emission_process = LearningHmmFilterEmissionProcess(self._config)
        self._transition_process = LearningHmmFilterTransitionProcess(
            self._config
        )

        # Initialize the estimated next state, and next observation for the inference mode
        self._init_batch_size(batch_size=1)

    def _init_batch_size(
        self, batch_size: int, is_initialized: bool = False
    ) -> None:
        self._is_initialized = is_initialized
        self._batch_size = batch_size
        with torch.no_grad():
            self._estimated_state = (
                torch.ones(
                    (self._batch_size, self._state_dim), dtype=torch.float64
                )
                / self._state_dim
            )  # (batch_size, state_dim)
            self._predicted_next_state = (
                torch.ones(
                    (self._batch_size, self._state_dim), dtype=torch.float64
                )
                / self._state_dim
            )  # (batch_size, state_dim)
            self._predicted_next_observation_probability: torch.Tensor = (
                self._emission_process(
                    self._predicted_next_state,  # (batch_size, state_dim)
                )
            )  # (batch_size, discrete_observation_dim)
            self._estimation = self._estimate()
            self._estimation_shape = tuple(self._estimation.shape[1:])

    estimated_state = BatchTensorReadOnlyDescriptor(
        "_batch_size", "_state_dim"
    )
    predicted_next_state = BatchTensorReadOnlyDescriptor(
        "_batch_size", "_state_dim"
    )
    predicted_next_observation_probability = BatchTensorReadOnlyDescriptor(
        "_batch_size", "_discrete_observation_dim"
    )
    estimation = BatchTensorReadOnlyDescriptor(
        "_batch_size", "*_estimation_shape"
    )

    state_dim = ReadOnlyDescriptor[int]()
    discrete_observation_dim = ReadOnlyDescriptor[int]()
    layer_dim = ReadOnlyDescriptor[int]()
    estimation_shape = ReadOnlyDescriptor[tuple]()
    batch_size = ReadOnlyDescriptor[int]()

    @property
    def emission_matrix(self) -> torch.Tensor:
        return self._emission_process.emission_matrix.detach()

    def forward(self, observation_trajectory: torch.Tensor) -> torch.Tensor:
        """
        forward method for the `LearningHmmFilter` class

        Arguments:
        ----------
            observation_trajectory : torch.Tensor
                shape (batch_size, horizon)

        Returns:
        --------
            predicted_next_observation_log_probability_trajectory : torch.Tensor
                shape (batch_size, discrete_observation_dim, horizon)
        """

        _, predicted_next_state_trajectory, emission_matrix = self._forward(
            observation_trajectory
        )  # (batch_size, horizon, discrete_observation_dim)

        predicted_next_observation_trajectory = self._emission_process(
            predicted_next_state_trajectory,  # (batch_size, horizon, state_dim)
            emission_matrix,  # (state_dim, observation_dim)
        )  # (batch_size, horizon, observation_dim)

        predicted_next_observation_log_probability_trajectory = torch.moveaxis(
            torch.log(predicted_next_observation_trajectory), 1, 2
        )  # (batch_size, discrete_observation_dim, horizon)

        return predicted_next_observation_log_probability_trajectory

    def _forward(
        self,
        observation_trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        predicted_next_state_trajectory: torch.Tensor
            shape (batch_size, horizon, state_dim)
        emission_matrix: torch.Tensor
            shape (state_dim, discrete_observation_dim)
        """

        # Get emission_matrix
        emission_matrix = (
            self._emission_process.emission_matrix
        )  # (state_dim, discrete_observation_dim)

        # Get emission based on each observation in the trajectory
        input_state_trajectory = torch.moveaxis(
            emission_matrix[:, observation_trajectory], 0, 2
        )  # (batch_size, horizon, state_dim)

        (
            estimated_state_trajectory,  # (batch_size, horizon, state_dim)
            predicted_next_state_trajectory,  # (batch_size, horizon, state_dim)
        ) = self._transition_process(input_state_trajectory)

        return (
            estimated_state_trajectory,
            predicted_next_state_trajectory,
            emission_matrix,
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

    def set_estimation_option(
        self,
        estimation_option: LearningHmmFilterEstimationOption,
    ) -> None:
        """
        Set the estimation option for the `LearningHiddenMarkovModelFilter` class.

        Arguments
        ---------
        estimation_option : LearningHiddenMarkovModelFilterEstimationOption
            The option for the estimation.
        """
        self._config.estimation_option = estimation_option
        self._init_batch_size(batch_size=self._batch_size)

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

        estimated_state_trajectory, predicted_next_state_trajectory, _ = (
            self._forward(
                self._device_manager.load_data(observation_trajectory)
            )
        )

        self._estimated_state = estimated_state_trajectory[
            :, -1, :
        ]  # (batch_size, state_dim)
        self._predicted_next_state = predicted_next_state_trajectory[
            :, -1, :
        ]  # (batch_size, state_dim)
        self._predicted_next_observation_probability = self._emission_process(
            self._predicted_next_state,  # (batch_size, state_dim)
        )  # (batch_size, discrete_observation_dim)

    @torch.inference_mode()
    def estimate(self) -> torch.Tensor:
        """
        Compute the estimation. This method should be called after the `update` method.

        Returns
        -------
        estimation: torch.Tensor
            Based on the `estimation_option` in the configuration, the chosen estimation will be returned.
        """
        self._estimation = self._estimate()
        return self.estimation

    @torch.inference_mode()
    def _estimate(self) -> torch.Tensor:
        match self._config.estimation_option:
            case (
                LearningHmmFilterEstimationOption.PREDICTED_NEXT_OBSERVATION_PROBABILITY_OVER_LAYERS
            ):
                estimation: torch.Tensor = self._emission_process(
                    self._transition_process.predicted_next_state_over_layers
                )
                if self._batch_size == 1:
                    estimation = estimation.unsqueeze(0)
            case (
                LearningHmmFilterEstimationOption.PREDICTED_NEXT_STATE_OVER_LAYERS
            ):
                estimation = (
                    self._transition_process.predicted_next_state_over_layers
                )
                if self._batch_size == 1:
                    estimation = estimation.unsqueeze(0)
            case LearningHmmFilterEstimationOption.ESTIMATED_STATE:
                estimation = self._estimated_state
            case LearningHmmFilterEstimationOption.PREDICTED_NEXT_STATE:
                estimation = self._predicted_next_state
            case (
                LearningHmmFilterEstimationOption.PREDICTED_NEXT_OBSERVATION_PROBABILITY
            ):
                estimation = self._predicted_next_observation_probability
            case _ as _invalid_estimation_option:
                assert_never(_invalid_estimation_option)
        return estimation

    @torch.inference_mode()
    def predict(self) -> torch.Tensor:
        """
        Predict the next observation.

        Returns
        -------
        predicted_observation: torch.Tensor
            shape = (batch_size, 1) or (1,) if batch_size is 1
        """
        predicted_next_observation = torch.multinomial(
            self.predicted_next_observation_probability,
            1,
            replacement=True,
        )  # (batch_size, 1) or (1,) if batch_size is 1
        return predicted_next_observation.detach()


class LearningHmmFilterProcess(BaseLearningProcess):
    def _evaluate_one_batch(
        self, data_batch: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        (
            observation_trajectory,
            next_observation_trajectory,
        ) = HmmObservationDataset.from_batch(
            data_batch
        )  # (batch_size, max_length), (batch_size, max_length)
        predicted_next_observation_log_probability_trajectory = self._model(
            observation_trajectory=observation_trajectory
        )  # (batch_size, discrete_observation_dim, max_length)
        _loss: torch.Tensor = self._loss_function(
            predicted_next_observation_log_probability_trajectory,
            next_observation_trajectory,  # (batch_size, max_length)
        )
        return _loss
