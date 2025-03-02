from typing import Optional, Tuple, assert_never, cast

import torch
from torch import nn

from ss.estimation.filtering.hmm.learning.module import config as Config
from ss.utility.learning.module import BaseLearningModule
from ss.utility.learning.parameter.probability import (
    ProbabilityParameter,
)
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class BaseTransitionBlock(BaseLearningModule[Config.TransitionBlockConfig]):
    def __init__(
        self,
        config: Config.TransitionBlockConfig,
        filter_config: Config.FilterConfig,
        block_id: int,
    ) -> None:
        super().__init__(config)
        self._state_dim = filter_config.state_dim
        self._block_id = block_id

        self._initial_state_parameter = ProbabilityParameter(
            self._config.initial_state.probability_parameter,
            (self._state_dim,),
        )

        self._is_initialized = False
        self._estimated_previous_state = (
            torch.ones(self._state_dim) / self._state_dim
        ).repeat(
            1, 1
        )  # (batch_size, state_dim)
        self._matrix_parameter: ProbabilityParameter

    @property
    def id(self) -> int:
        return self._block_id

    def get_estimated_previous_state(
        self, batch_size: int = 1
    ) -> torch.Tensor:
        if not self._inference:
            self._is_initialized = False
            estimated_previous_state = self.initial_state.repeat(batch_size, 1)
            return estimated_previous_state
        if not self._is_initialized:
            self._is_initialized = True
            self._estimated_previous_state = self.initial_state.repeat(
                batch_size, 1
            )
        return self._estimated_previous_state

    def reset(self) -> None:
        self._is_initialized = False

    @property
    def initial_state_parameter(self) -> ProbabilityParameter:
        return self._initial_state_parameter

    @property
    def initial_state(self) -> torch.Tensor:
        return cast(torch.Tensor, self._initial_state_parameter())

    @initial_state.setter
    def initial_state(self, initial_state: torch.Tensor) -> None:
        self._initial_state_parameter.set_value(initial_state)

    @property
    def matrix_parameter(self) -> ProbabilityParameter:
        return self._matrix_parameter

    @property
    def matrix(self) -> torch.Tensor:
        raise NotImplementedError

    @matrix.setter
    def matrix(self, matrix: torch.Tensor) -> None:
        raise NotImplementedError

    def _prediction_step(
        self,
        previous_estimated_state: torch.Tensor,
        transition_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if transition_matrix is None:
            transition_matrix = self.matrix
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
            device=self._device_manager.device,
        )

        predicted_next_state_trajectory = torch.zeros(
            (batch_size, horizon, self._state_dim),
            device=self._device_manager.device,
        )

        transition_matrix = self.matrix  # (state_dim, state_dim)

        estimated_previous_state = self.get_estimated_previous_state(
            batch_size
        )  # (batch_size, state_dim)

        # prediction step based on model process (predicted probability)
        predicted_state = self._prediction_step(
            estimated_previous_state,
            (
                torch.eye(
                    self._state_dim,
                    device=self._device_manager.device,
                )
                if self._config.skip_first_transition
                else transition_matrix
            ),
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

        if self._inference:
            self._estimated_previous_state = (
                predicted_state
                if self._config.skip_first_transition
                else estimated_previous_state
            )

        return estimated_state_trajectory, predicted_next_state_trajectory

    @classmethod
    def create(
        cls,
        config: Config.TransitionBlockConfig,
        filter_config: Config.FilterConfig,
        block_id: int,
    ) -> "BaseTransitionBlock":
        match config.option:
            case Config.TransitionBlockConfig.Option.FULL_MATRIX:
                return TransitionFullMatrix(config, filter_config, block_id)
            case Config.TransitionBlockConfig.Option.SPATIAL_INVARIANT_MATRIX:
                return TransitionSpatialInvariantMatrix(
                    config,
                    filter_config,
                    block_id,
                )
            case Config.TransitionBlockConfig.Option.IID:
                return TransitionIID(config, filter_config, block_id)
            case _ as _invalid_block_option:
                assert_never(_invalid_block_option)


class TransitionFullMatrix(BaseTransitionBlock):
    def __init__(
        self,
        config: Config.TransitionBlockConfig,
        filter_config: Config.FilterConfig,
        block_id: int,
    ) -> None:
        super().__init__(config, filter_config, block_id)
        self._matrix_parameter = ProbabilityParameter(
            self._config.matrix.probability_parameter,
            (self._state_dim, self._state_dim),
        )

    @property
    def matrix(self) -> torch.Tensor:
        matrix: torch.Tensor = self._matrix_parameter()
        return matrix

    @matrix.setter
    def matrix(self, matrix: torch.Tensor) -> None:
        self._matrix_parameter.set_value(matrix)


class TransitionSpatialInvariantMatrix(BaseTransitionBlock):
    def __init__(
        self,
        config: Config.TransitionBlockConfig,
        filter_config: Config.FilterConfig,
        block_id: int,
    ) -> None:
        super().__init__(config, filter_config, block_id)
        self._matrix_parameter = ProbabilityParameter(
            self._config.matrix.probability_parameter,
            (self._state_dim,),
        )

        if self._config.matrix.initial_state_binding:
            self._matrix_parameter.binding(
                self._initial_state_parameter.parameter
            )
            self._matrix_parameter.transformer.temperature.binding(
                self._initial_state_parameter.transformer.temperature.parameter
            )

    @property
    def matrix(self) -> torch.Tensor:
        row_probability: torch.Tensor = self._matrix_parameter()
        matrix = torch.empty(
            (self._state_dim, self._state_dim),
            device=row_probability.device,
        )
        matrix[0, :] = row_probability
        for i in range(1, self._state_dim):
            matrix[i, :] = torch.roll(matrix[i - 1, :], shifts=1)
        return matrix

    @matrix.setter
    def matrix(self, matrix: torch.Tensor) -> None:
        for i in range(1, self._state_dim):
            if not torch.allclose(
                matrix[i, :],
                torch.roll(matrix[i - 1, :], shifts=1),
            ):
                raise ValueError(
                    "The matrix is not spatial invariant. "
                    "The matrix must have the same shifted row probability."
                )
        row_probability = matrix[0, :]
        self._matrix_parameter.set_value(row_probability)


class TransitionIID(BaseTransitionBlock):
    def __init__(
        self,
        config: Config.TransitionBlockConfig,
        filter_config: Config.FilterConfig,
        block_id: int,
    ) -> None:
        super().__init__(config, filter_config, block_id)

        if self._config.matrix.initial_state_binding is not True:
            self._config.matrix.initial_state_binding = True
            logger.warning(
                "transition IID matrix requires initial state binding. "
                "Automatically set initial state binding to True in the configuration."
            )
        self._matrix_parameter = ProbabilityParameter(
            self._config.matrix.probability_parameter,
            (self._state_dim,),
        )
        self._matrix_parameter.binding(self._initial_state_parameter.parameter)
        self._matrix_parameter.transformer.temperature.binding(
            self._initial_state_parameter.transformer.temperature.parameter
        )

    @property
    def matrix(self) -> torch.Tensor:
        row_probability: torch.Tensor = self._matrix_parameter()
        matrix = torch.empty(
            (self._state_dim, self._state_dim),
            device=row_probability.device,
        )
        matrix[0, :] = row_probability
        for i in range(1, self._state_dim):
            matrix[i, :] = matrix[i - 1, :]
        return matrix

    @matrix.setter
    def matrix(self, matrix: torch.Tensor) -> None:
        for i in range(1, self._state_dim):
            if not torch.allclose(
                matrix[i, :],
                matrix[i - 1, :],
            ):
                raise ValueError(
                    "The matrix is not IID. "
                    "The matrix must have the same row probability."
                )
        logger.warning(
            "Transition IID matrix is bound with initial state. "
            "Changing the matrix also modifies the initial state."
        )
        row_probability = matrix[0, :]
        self._matrix_parameter.set_value(row_probability)
