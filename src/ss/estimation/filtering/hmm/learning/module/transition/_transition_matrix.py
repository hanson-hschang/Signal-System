from typing import Optional, Tuple, assert_never, cast

import torch
from torch import nn

from ss.estimation.filtering.hmm.learning.module import config as Config
from ss.utility.learning.module import BaseLearningModule
from ss.utility.learning.module.dropout import Dropout
from ss.utility.learning.module.probability import Probability


class BaseLearningHmmFilterTransitionMatrix(
    BaseLearningModule[Config.LearningHmmFilterConfig]
):
    def __init__(
        self,
        config: Config.LearningHmmFilterConfig,
        feature_id: int,
    ) -> None:
        super().__init__(config)
        self._feature_id = feature_id
        self._state_dim = self._config.filter.state_dim

        self._dropout = Dropout(self._config.dropout)

        self._initial_state_parameter = nn.Parameter(
            self._config.transition.initial_state.initializer.initialize(
                self._state_dim,
            )
        )  # (state_dim,)
        self._initial_state_probability = Probability.create(
            self._config.transition.initial_state.probability
        )

        self._is_initialized = False
        self._estimated_previous_state = (
            torch.ones(self._state_dim, dtype=torch.float64) / self._state_dim
        ).repeat(
            1, 1
        )  # (batch_size, state_dim)
        self._matrix_parameter: nn.Parameter

    def get_estimated_previous_state(
        self, batch_size: int = 1
    ) -> torch.Tensor:
        if not self._inference:
            self._is_initialized = False
            _estimated_previous_state = self._initial_state_probability(
                self._initial_state_parameter
            )
            return cast(torch.Tensor, _estimated_previous_state).repeat(
                batch_size, 1
            )
        if not self._is_initialized:
            self._is_initialized = True
            _estimated_previous_state = self._initial_state_probability(
                self._initial_state_parameter
            )
            self._estimated_previous_state = cast(
                torch.Tensor, _estimated_previous_state
            ).repeat(batch_size, 1)
        return self._estimated_previous_state

    def reset(self) -> None:
        self._is_initialized = False

    @property
    def initial_state_parameter(self) -> nn.Parameter:
        return self._initial_state_parameter

    @property
    def initial_state(self) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self._initial_state_probability(self._initial_state_parameter),
        )

    @property
    def matrix_parameter(self) -> nn.Parameter:
        return self._matrix_parameter

    @property
    def matrix(self) -> torch.Tensor:
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
            dtype=torch.float64,
            device=self._device_manager.device,
        )

        predicted_next_state_trajectory = torch.zeros(
            (batch_size, horizon, self._state_dim),
            dtype=torch.float64,
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
                    dtype=torch.float64,
                    device=self._device_manager.device,
                )
                if self._config.transition.matrix.skip_first_transition
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
                if self._config.transition.matrix.skip_first_transition
                else estimated_previous_state
            )

        return estimated_state_trajectory, predicted_next_state_trajectory

    @classmethod
    def create(
        cls, config: Config.LearningHmmFilterConfig, feature_id: int
    ) -> "BaseLearningHmmFilterTransitionMatrix":
        match config.transition.matrix.option:
            case Config.TransitionMatrixConfig.Option.FULL_MATRIX:
                return LearningHmmFilterTransitionFullMatrix(
                    config, feature_id
                )
            case Config.TransitionMatrixConfig.Option.SPATIAL_INVARIANT:
                return LearningHmmFilterTransitionSpatialInvariantMatrix(
                    config, feature_id
                )
            case Config.TransitionMatrixConfig.Option.IID:
                return LearningHmmFilterTransitionIIDMatrix(config, feature_id)
            case _ as _invalid_block_option:
                assert_never(_invalid_block_option)


class LearningHmmFilterTransitionFullMatrix(
    BaseLearningHmmFilterTransitionMatrix
):
    def __init__(
        self, config: Config.LearningHmmFilterConfig, feature_id: int
    ) -> None:
        super().__init__(config, feature_id)
        _weight = torch.empty(
            (self._state_dim, self._state_dim),
            dtype=torch.float64,
        )
        for i in range(self._state_dim):
            _weight[i, :] = (
                self._config.transition.matrix.initializer.initialize(
                    self._state_dim, i
                )
            )
        self._matrix_parameter = nn.Parameter(_weight)
        self._matrix_probability = Probability.create(
            self._config.transition.matrix.probability
        )
        # self._mask = torch.ones_like(self._weight)

    @property
    def matrix(self) -> torch.Tensor:
        matrix: torch.Tensor = self._matrix_probability(
            self._dropout(self._matrix_parameter)
        )
        return matrix

        # mask = self._dropout(self._mask).to(device=self._weight.device)
        # extended_weight = torch.cat(
        #     [
        #         self._weight,
        #         torch.ones(
        #             (self._state_dim, 1),
        #             dtype=self._weight.dtype,
        #             device=self._weight.device,
        #         ),
        #     ],
        #     dim=1,
        # )
        # row_norms = (
        #     torch.norm(extended_weight, p=2, dim=1)
        #     .unsqueeze(dim=1)
        #     .expand(self._state_dim, self._state_dim)
        # )
        # weight = nn.functional.softmax(
        #     mask * self._weight
        #     + (1 - mask) * row_norms * self._config.dropout.value.log_zero_scale,
        #     dim=1,
        # )

        # return nn.functional.softmax(weight, dim=1)


class LearningHmmFilterTransitionSpatialInvariantMatrix(
    BaseLearningHmmFilterTransitionMatrix
):
    def __init__(
        self,
        config: Config.LearningHmmFilterConfig,
        feature_id: int,
    ) -> None:
        super().__init__(config, feature_id)
        _weight = self._config.transition.matrix.initializer.initialize(
            self._state_dim,
        )
        if self._config.transition.matrix.initial_state_binding:
            _weight = self._initial_state_parameter
        self._matrix_parameter = nn.Parameter(_weight)
        self._matrix_probability = Probability.create(
            self._config.transition.matrix.probability
        )
        # self._mask = torch.ones_like(self._weight)

    @property
    def matrix(self) -> torch.Tensor:
        matrix = torch.empty(
            (self._state_dim, self._state_dim),
            dtype=torch.float64,
            device=self._matrix_parameter.device,
        )
        matrix[0, :] = self._matrix_probability(
            self._dropout(self._matrix_parameter),
        )
        for i in range(1, self._state_dim):
            matrix[i, :] = torch.roll(matrix[i - 1, :], shifts=1)
        return matrix

        # mask = self._dropout(self._mask).to(device=self._weight.device)
        # extended_weight = torch.cat(
        #     [
        #         self._weight,
        #         torch.tensor(
        #             [1], dtype=self._weight.dtype, device=self._weight.device
        #         ),
        #     ],
        #     dim=0,
        # )
        # norm = (
        #     torch.norm(extended_weight, p=2, dim=0)
        #     .unsqueeze(dim=0)
        #     .expand(self._state_dim)
        # )
        # matrix[0, :] = nn.functional.softmax(
        #     mask * self._weight
        #     + (1 - mask) * norm * self._config.dropout.value.log_zero_scale,
        #     dim=0,
        # )
        # for i in range(1, self._state_dim):
        #     matrix[i, :] = torch.roll(matrix[i - 1, :], shifts=1)
        # return matrix


class LearningHmmFilterTransitionIIDMatrix(
    BaseLearningHmmFilterTransitionMatrix
):
    def __init__(
        self,
        config: Config.LearningHmmFilterConfig,
        feature_id: int,
    ) -> None:
        super().__init__(config, feature_id)
        # _weight = self._config.transition.matrix.initializer.initialize(
        #     self._state_dim,
        # )
        # if self._config.transition.matrix.initial_state_binding:
        self._config.transition.matrix.initial_state_binding = True
        _weight = self._initial_state_parameter
        self._matrix_parameter = nn.Parameter(_weight)
        self._matrix_probability = Probability.create(
            self._config.transition.matrix.probability
        )

    @property
    def matrix(self) -> torch.Tensor:
        matrix = torch.empty(
            (self._state_dim, self._state_dim),
            dtype=torch.float64,
            device=self._matrix_parameter.device,
        )
        matrix[0, :] = self._matrix_probability(
            self._dropout(self._matrix_parameter)
        )
        for i in range(1, self._state_dim):
            matrix[i, :] = matrix[i - 1, :]
        return matrix
