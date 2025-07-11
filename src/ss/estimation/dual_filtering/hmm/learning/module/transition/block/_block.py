from typing import Generic, List, Tuple, TypeVar, assert_never

import torch

from ss.estimation.dual_filtering.hmm.learning.module.filter.config import (
    DualFilterConfig,
)
from ss.estimation.dual_filtering.hmm.learning.module.transition.block.config import (
    DualTransitionBlockConfig,
)
from ss.estimation.dual_filtering.hmm.learning.module.transition.step import (
    DualTransitionStepMixin,
)
from ss.utility.descriptor import ReadOnlyDescriptor
from ss.utility.learning.module import BaseLearningModule
from ss.utility.learning.parameter.probability import ProbabilityParameter
from ss.utility.learning.parameter.probability.config import (
    ProbabilityParameterConfig,
)
from ss.utility.learning.parameter.transformer import T
from ss.utility.learning.parameter.transformer.config import TC
from ss.utility.learning.parameter.transformer.softmax import (
    SoftmaxTransformer,
)
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class BaseDualTransitionBlock(
    DualTransitionStepMixin[T, TC],
    BaseLearningModule[DualTransitionBlockConfig[TC]],
    Generic[T, TC],
):
    def __init__(
        self,
        config: DualTransitionBlockConfig[TC],
        filter_config: DualFilterConfig,
        block_id: int,
    ) -> None:
        super().__init__(
            config=config,
            step_config=config.step,
            filter_config=filter_config,
        )
        self._id = block_id

    id = ReadOnlyDescriptor[int]()

    def _get_internal_estimated_state(
        self, batch_size: int = 1
    ) -> torch.Tensor:
        if self._inference:
            return super()._get_internal_estimated_state(batch_size)
        self._is_initialized = False
        estimated_state = self.initial_state.repeat(batch_size, 1)
        return estimated_state

    def _set_internal_estimated_state(
        self, estimated_state: torch.Tensor, predicted_next_state: torch.Tensor
    ) -> None:
        if self._inference:
            super()._set_internal_estimated_state(
                estimated_state, predicted_next_state
            )

    def forward(
        self,
        likelihood_state_trajectory: torch.Tensor,
        # emission_difference_trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        estimated_state_trajectory, control_trajectories = self.process(
            likelihood_state_trajectory
        )

        return estimated_state_trajectory, control_trajectories

    @classmethod
    def create(
        cls,
        config: DualTransitionBlockConfig,
        filter_config: DualFilterConfig,
        block_id: int,
    ) -> "BaseDualTransitionBlock[T, TC]":
        match config.option:
            case DualTransitionBlockConfig.Option.FULL_MATRIX:
                return DualTransitionFullMatrix[T, TC](
                    config, filter_config, block_id
                )
            case DualTransitionBlockConfig.Option.SPATIAL_INVARIANT_MATRIX:
                return DualTransitionSpatialInvariantMatrix[T, TC](
                    config,
                    filter_config,
                    block_id,
                )
            case DualTransitionBlockConfig.Option.IID:
                return DualTransitionIID[T, TC](
                    config, filter_config, block_id
                )
            case _ as _invalid_block_option:
                assert_never(_invalid_block_option)


class DualTransitionFullMatrix(BaseDualTransitionBlock[T, TC], Generic[T, TC]):
    def __init__(
        self,
        config: DualTransitionBlockConfig[TC],
        filter_config: DualFilterConfig,
        block_id: int,
    ) -> None:
        super().__init__(config, filter_config, block_id)
        self._matrix = ProbabilityParameter[T, TC](
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


class DualTransitionSpatialInvariantMatrix(
    BaseDualTransitionBlock[T, TC], Generic[T, TC]
):
    def __init__(
        self,
        config: DualTransitionBlockConfig,
        filter_config: DualFilterConfig,
        block_id: int,
    ) -> None:
        super().__init__(config, filter_config, block_id)
        self._matrix = ProbabilityParameter[T, TC](
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


class DualTransitionIID(BaseDualTransitionBlock[T, TC], Generic[T, TC]):
    def __init__(
        self,
        config: DualTransitionBlockConfig,
        filter_config: DualFilterConfig,
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
