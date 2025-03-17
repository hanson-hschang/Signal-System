from typing import Callable, Generic, List, Tuple, TypeVar, cast

import torch
from torch import nn

from ss.estimation.filtering.hmm.learning.module.filter.config import (
    FilterConfig,
)
from ss.estimation.filtering.hmm.learning.module.transition.block import (
    BaseTransitionBlock,
)

# from ss.estimation.filtering.hmm.learning.module import config as Config
from ss.estimation.filtering.hmm.learning.module.transition.layer.config import (
    TransitionLayerConfig,
)
from ss.estimation.filtering.hmm.learning.module.transition.step import (
    TransitionStepMixin,
)
from ss.utility.descriptor import ReadOnlyDescriptor
from ss.utility.learning.module import BaseLearningModule, reset_module
from ss.utility.learning.parameter.initializer.normal_distribution import (
    NormalDistributionInitializer,
)
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

# TC = TypeVar("TC", bound=TransformerConfig)
# T = TypeVar("T", bound=Transformer)


@torch.compile
def weighted_sum(
    base_tensor: torch.Tensor,
    incoming_tensor: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return base_tensor + incoming_tensor * weight


class TransitionLayer(
    TransitionStepMixin[T, TC],
    BaseLearningModule[TransitionLayerConfig[TC]],
    Generic[T, TC],
):
    def __init__(
        self,
        config: TransitionLayerConfig[TC],
        filter_config: FilterConfig,
        layer_id: int,
    ) -> None:
        super().__init__(
            config=config,
            step_config=config.step,
            filter_config=filter_config,
            # initial_state_config=config.initial_state,
            # state_dim=filter_config.state_dim,
            # skip_first_transition=config.skip_first_transition,
        )
        # self._state_dim = filter_config.state_dim
        self._id = layer_id
        self._block_dim = self._config.block_dim

        self._blocks = nn.ModuleList()
        for block_id in range(self._block_dim):
            block_config = self._config.blocks[block_id]
            block_config.step.skip_first = self._config.step.skip_first
            self._blocks.append(
                BaseTransitionBlock[T, TC].create(
                    block_config,
                    filter_config,
                    block_id,
                )
            )

        self._block_state_binding = self._config.block_state_binding

        # self._initial_state: ProbabilityParameter

        self._forward: Callable[
            [torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
        ]
        if self._block_state_binding:
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
            for block in self._blocks:
                cast(
                    BaseTransitionBlock[T, TC], block
                ).initial_state_parameter.bind_with(self._initial_state)
            self._forward = self._forward_bound_initial_state
        else:
            del self._initial_state
            self._forward = self._forward_unbound_initial_state

        if self._block_dim == 1:
            self._config.coefficient.probability_parameter.require_training = (
                False
            )
            self._config.coefficient.probability_parameter.initializer = (
                NormalDistributionInitializer.basic_config(
                    mean=0.0,
                    std=0.0,
                )
            )
        self._coefficient = ProbabilityParameter[T, TC](
            self._config.coefficient.probability_parameter,
            (
                (self._state_dim, self._block_dim)
                if self._block_state_binding
                else (self._block_dim,)
            ),
        )

    id = ReadOnlyDescriptor[int]()
    block_dim = ReadOnlyDescriptor[int]()
    block_state_binding = ReadOnlyDescriptor[bool]()

    @property
    def coefficient_parameter(
        self,
    ) -> ProbabilityParameter[T, TC]:
        return self._coefficient

    @property
    def coefficient(self) -> torch.Tensor:
        coefficient: torch.Tensor = self._coefficient()
        return coefficient

    @coefficient.setter
    def coefficient(self, coefficient: torch.Tensor) -> None:
        self._coefficient.set_value(coefficient)

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

    @property
    def blocks(self) -> List[BaseTransitionBlock[T, TC]]:
        return [
            cast(BaseTransitionBlock[T, TC], block) for block in self._blocks
        ]

    @property
    def matrix(self) -> torch.Tensor:
        if not self._block_state_binding:
            raise AttributeError(
                "The matrix attribute is only available when block_initial_state_binding is True."
            )
        coefficient = self.coefficient
        transition_matrix = torch.zeros(
            (self._state_dim, self._state_dim),
            device=coefficient.device,
        )
        for i, block in enumerate(self._blocks):
            transition_matrix += (
                cast(BaseTransitionBlock[T, TC], block).matrix
                * coefficient[:, i : i + 1]
            )
        return transition_matrix

    @matrix.setter
    def matrix(self, matrix: torch.Tensor) -> None:
        if not self._block_state_binding:
            raise AttributeError(
                "The matrix attribute is only available when block_initial_state_binding is True."
            )
        raise AttributeError("The matrix attribute is read-only.")

    def _get_internal_estimated_state(
        self, batch_size: int = 1
    ) -> torch.Tensor:
        if self._inference:
            return super()._get_internal_estimated_state(batch_size)
        self._is_initialized = False
        estimated_state = self.initial_state.repeat(batch_size, 1)
        return estimated_state
        # if not self._inference:
        #     self._is_initialized = False
        #     estimated_state = self.initial_state.repeat(batch_size, 1)
        #     return estimated_state
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

    def _forward_bound_initial_state(
        self, likelihood_state_trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            estimated_state_trajectory,
            predicted_next_state_trajectory,
        ) = self.process(likelihood_state_trajectory)

        return (
            estimated_state_trajectory,
            predicted_next_state_trajectory,
        )

    def _forward_unbound_initial_state(
        self, likelihood_state_trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        coefficient = self.coefficient.unsqueeze(dim=0).unsqueeze(dim=0)
        # (1, 1, block_dim)

        estimated_state_trajectory = torch.zeros_like(
            likelihood_state_trajectory,
            device=likelihood_state_trajectory.device,
        )  # (batch_size, horizon, state_dim)
        predicted_next_state_trajectory = torch.zeros_like(
            likelihood_state_trajectory,
            device=likelihood_state_trajectory.device,
        )  # (batch_size, horizon, state_dim)

        for b, block in enumerate(self._blocks):
            _estimated_state_trajectory, _predicted_next_state_trajectory = (
                block(likelihood_state_trajectory)
            )  # (batch_size, horizon, state_dim), (batch_size, horizon, state_dim)
            # estimated_state_trajectory += (
            #     _estimated_state_trajectory * coefficient[:, :, b]
            # )
            # predicted_next_state_trajectory += (
            #     _predicted_next_state_trajectory * coefficient[:, :, b]
            # )
            estimated_state_trajectory = weighted_sum(
                estimated_state_trajectory,
                _estimated_state_trajectory,
                coefficient[:, :, b],
            )
            predicted_next_state_trajectory = weighted_sum(
                predicted_next_state_trajectory,
                _predicted_next_state_trajectory,
                coefficient[:, :, b],
            )

        return (
            estimated_state_trajectory,
            predicted_next_state_trajectory,
        )

    def forward(
        self, likelihood_state_trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        (
            estimated_state_trajectory,
            predicted_next_state_trajectory,
        ) = self._forward(likelihood_state_trajectory)
        # coefficient = self.coefficient.unsqueeze(dim=0).unsqueeze(dim=0)
        # # (1, 1, state_dim, block_dim) or (1, 1, block_dim)

        # average_estimated_state_trajectory = torch.zeros_like(
        #     input_state_trajectory
        # )  # (batch_size, horizon, state_dim)
        # average_predicted_next_state_trajectory = torch.zeros_like(
        #     input_state_trajectory
        # )  # (batch_size, horizon, state_dim)

        # for b, block in enumerate(self._blocks):
        #     estimated_state_trajectory, predicted_next_state_trajectory = (
        #         block(input_state_trajectory)
        #     )  # (batch_size, horizon, state_dim), (batch_size, horizon, state_dim)
        #     average_estimated_state_trajectory += (
        #         estimated_state_trajectory * coefficient[:, :, b]
        #     )
        #     average_predicted_next_state_trajectory += (
        #         predicted_next_state_trajectory * coefficient[:, :, b]
        #     )
        return (
            estimated_state_trajectory,
            predicted_next_state_trajectory,
        )

    def reset(self) -> None:
        for block in self._blocks:
            reset_module(block)
        if self._block_state_binding:
            super().reset()
