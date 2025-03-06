from typing import Generic, Tuple, TypeVar, assert_never

import torch

from ss.estimation.filtering.hmm.learning.module import config as Config
from ss.estimation.filtering.hmm.learning.module.transition.step import (
    TransitionStepMixin,
)
from ss.utility.descriptor import ReadOnlyDescriptor
from ss.utility.learning.module import BaseLearningModule
from ss.utility.learning.parameter.probability import ProbabilityParameter
from ss.utility.learning.parameter.probability.config import (
    ProbabilityParameterConfig,
)
from ss.utility.learning.parameter.transformer import Transformer
from ss.utility.learning.parameter.transformer.softmax import (
    SoftmaxTransformer,
)
from ss.utility.logging import Logging

T = TypeVar("T", bound=Transformer, default=SoftmaxTransformer)

logger = Logging.get_logger(__name__)


class BaseTransitionBlock(
    TransitionStepMixin[T],
    BaseLearningModule[Config.TransitionBlockConfig],
    Generic[T],
):
    def __init__(
        self,
        config: Config.TransitionBlockConfig,
        filter_config: Config.FilterConfig,
        block_id: int,
    ) -> None:
        super().__init__(
            config=config,
            initial_state_config=config.initial_state,
            state_dim=filter_config.state_dim,
            skip_first_transition=config.skip_first_transition,
        )
        self._id = block_id
        # self._state_dim = filter_config.state_dim

        # self._initial_state = ProbabilityParameter(
        #     self._config.initial_state.probability_parameter,
        #     (self._state_dim,),
        # )

        # self._is_initialized = False
        # self._estimated_state = (
        #     torch.ones(self._state_dim) / self._state_dim
        # ).repeat(
        #     1, 1
        # )  # (batch_size, state_dim)
        # self._matrix: ProbabilityParameter

    # @property
    # def id(self) -> int:
    #     return self._id
    id = ReadOnlyDescriptor[int]()

    # @property
    # def estimated_state(self) -> torch.Tensor:
    #     batch_size, _ = self._estimated_state.shape
    #     if batch_size == 1:
    #         return self._estimated_state.squeeze(0)
    #     return self._estimated_state

    # @property
    # def predicted_next_state(self) -> torch.Tensor:
    #     transition_matrix = self.matrix  # (state_dim, state_dim)
    #     return TransitionStep.prediction(
    #         self.estimated_state, transition_matrix
    #     )

    def _get_internal_estimated_state(
        self, batch_size: int = 1
    ) -> torch.Tensor:
        if self._inference:
            return super()._get_internal_estimated_state(batch_size)
        self._is_initialized = False
        estimated_state = self.initial_state.repeat(batch_size, 1)
        return estimated_state
        # if not self._is_initialized:
        #     self._is_initialized = True
        #     self._estimated_state = self.initial_state.repeat(batch_size, 1)
        # return self._estimated_state

    def _set_internal_estimated_state(
        self, estimated_state: torch.Tensor, predicted_next_state: torch.Tensor
    ) -> None:
        if self._inference:
            super()._set_internal_estimated_state(
                estimated_state, predicted_next_state
            )

    # def reset(self) -> None:
    #     self._is_initialized = False
    #     self.get_estimated_state()
    #     self._is_initialized = False

    # @property
    # def initial_state_parameter(self) -> ProbabilityParameter:
    #     return self._initial_state

    # @property
    # def initial_state(self) -> torch.Tensor:
    #     initial_state: torch.Tensor = self._initial_state()
    #     return initial_state

    # @initial_state.setter
    # def initial_state(self, initial_state: torch.Tensor) -> None:
    #     self._initial_state.set_value(initial_state)

    # @property
    # def matrix_parameter(self) -> ProbabilityParameter:
    #     return self._matrix

    # @property
    # def matrix(self) -> torch.Tensor:
    #     raise NotImplementedError

    # @matrix.setter
    # def matrix(self, matrix: torch.Tensor) -> None:
    #     raise NotImplementedError

    def forward(
        self, likelihood_state_trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        (
            estimated_state_trajectory,
            predicted_next_state_trajectory,
        ) = self.process(likelihood_state_trajectory)

        return estimated_state_trajectory, predicted_next_state_trajectory

        # if self._inference:
        #     self._estimated_state = (
        #         predicted_next_state_trajectory[:, -1, :]
        #         if self._skip_first_transition
        #         else estimated_state_trajectory[:, -1, :]
        #     )

        # batch_size, horizon, _ = likelihood_state_trajectory.shape
        # # (batch_size, horizon, state_dim)

        # estimated_state_trajectory = torch.zeros(
        #     (batch_size, horizon, self._state_dim),
        #     device=self._device_manager.device,
        # )

        # predicted_next_state_trajectory = torch.zeros(
        #     (batch_size, horizon, self._state_dim),
        #     device=self._device_manager.device,
        # )

        # transition_matrix = self.matrix  # (state_dim, state_dim)

        # estimated_state = self.get_estimated_state(
        #     batch_size
        # )  # (batch_size, state_dim)

        # # prediction step based on model process (predicted probability)
        # predicted_next_state = TransitionStep.prediction(
        #     estimated_state,
        #     (
        #         torch.eye(
        #             self._state_dim,
        #             device=self._device_manager.device,
        #         )
        #         if self._config.skip_first_transition
        #         else transition_matrix
        #     ),
        # )  # (batch_size, state_dim)

        # for k in range(horizon):
        #     likelihood_state = likelihood_state_trajectory[
        #         :, k, :
        #     ]  # (batch_size, state_dim)

        #     # update step based on input_state (conditional probability)
        #     estimated_state = TransitionStep.update(
        #         predicted_next_state, likelihood_state
        #     )  # (batch_size, state_dim)

        #     # prediction step based on model process (predicted probability)
        #     predicted_next_state = TransitionStep.prediction(
        #         estimated_state, transition_matrix
        #     )  # (batch_size, state_dim)

        #     estimated_state_trajectory[:, k, :] = estimated_state
        #     predicted_next_state_trajectory[:, k, :] = predicted_next_state

        # if self._inference:
        #     self._estimated_state = (
        #         predicted_next_state
        #         if self._config.skip_first_transition
        #         else estimated_state
        #     )

        # return estimated_state_trajectory, predicted_next_state_trajectory

    @classmethod
    def create(
        cls,
        config: Config.TransitionBlockConfig,
        filter_config: Config.FilterConfig,
        block_id: int,
    ) -> "BaseTransitionBlock[T]":
        match config.option:
            case Config.TransitionBlockConfig.Option.FULL_MATRIX:
                return TransitionFullMatrix[T](config, filter_config, block_id)
            case Config.TransitionBlockConfig.Option.SPATIAL_INVARIANT_MATRIX:
                return TransitionSpatialInvariantMatrix[T](
                    config,
                    filter_config,
                    block_id,
                )
            case Config.TransitionBlockConfig.Option.IID:
                return TransitionIID[T](config, filter_config, block_id)
            case _ as _invalid_block_option:
                assert_never(_invalid_block_option)


class TransitionFullMatrix(BaseTransitionBlock[T], Generic[T]):
    def __init__(
        self,
        config: Config.TransitionBlockConfig,
        filter_config: Config.FilterConfig,
        block_id: int,
    ) -> None:
        super().__init__(config, filter_config, block_id)
        self._matrix = ProbabilityParameter(
            self._config.matrix.probability_parameter,
            (self._state_dim, self._state_dim),
        )

    @property
    def matrix(self) -> torch.Tensor:
        matrix: torch.Tensor = self._matrix()
        return matrix

    @matrix.setter
    def matrix(self, matrix: torch.Tensor) -> None:
        self._matrix.set_value(matrix)


class TransitionSpatialInvariantMatrix(BaseTransitionBlock[T], Generic[T]):
    def __init__(
        self,
        config: Config.TransitionBlockConfig,
        filter_config: Config.FilterConfig,
        block_id: int,
    ) -> None:
        super().__init__(config, filter_config, block_id)
        self._matrix = ProbabilityParameter(
            self._config.matrix.probability_parameter,
            (self._state_dim,),
        )

        if self._config.matrix.initial_state_binding:
            self._matrix.bind_with(self._initial_state)
            # self._matrix.transformer.temperature_parameter.bind_with(
            #     self._initial_state.transformer.temperature_parameter
            # )

    @property
    def matrix(self) -> torch.Tensor:
        row_probability: torch.Tensor = self._matrix()
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
        self._matrix.set_value(row_probability)


class TransitionIID(BaseTransitionBlock[T], Generic[T]):
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
        self._matrix = ProbabilityParameter(
            self._config.matrix.probability_parameter,
            (self._state_dim,),
        )
        self._matrix.bind_with(self._initial_state)
        # self._matrix.transformer.temperature_parameter.bind_with(
        #     self._initial_state.transformer.temperature_parameter
        # )

    @property
    def matrix(self) -> torch.Tensor:
        row_probability: torch.Tensor = self._matrix()
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
        self._matrix.set_value(row_probability)
