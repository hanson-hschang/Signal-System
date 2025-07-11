from typing import Callable, Generic, List, Optional, Tuple, TypeVar, cast

import torch
from torch import nn

from ss.estimation.dual_filtering.hmm.learning.module.filter.config import (
    DualFilterConfig,
)
from ss.estimation.dual_filtering.hmm.learning.module.transition.block import (
    BaseDualTransitionBlock,
)
from ss.estimation.dual_filtering.hmm.learning.module.transition.layer.config import (
    DualTransitionLayerConfig,
)
from ss.estimation.dual_filtering.hmm.learning.module.transition.step import (
    DualTransitionStepMixin,
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


@torch.compile
def weighted_sum(
    base_tensor: torch.Tensor,
    incoming_tensor: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    state_dim = incoming_tensor.shape[-1]
    return base_tensor + incoming_tensor * weight.unsqueeze(-1).expand(
        *weight.shape, state_dim
    )


class DualTransitionLayer(
    DualTransitionStepMixin[T, TC],
    BaseLearningModule[DualTransitionLayerConfig[TC]],
    Generic[T, TC],
):
    def __init__(
        self,
        config: DualTransitionLayerConfig[TC],
        filter_config: DualFilterConfig,
        layer_id: int,
    ) -> None:
        super().__init__(
            config=config,
            step_config=config.step,
            filter_config=filter_config,
        )

        self._id = layer_id
        self._block_dim = self._config.block_dim

        self._blocks = nn.ModuleList()
        for block_id in range(self._block_dim):
            block_config = self._config.blocks[block_id]
            block_config.step.skip_first = self._config.step.skip_first
            self._blocks.append(
                BaseDualTransitionBlock[T, TC].create(
                    block_config,
                    filter_config,
                    block_id,
                )
            )

        self._block_state_binding = self._config.block_state_binding

        self._forward: Callable[
            [torch.Tensor, Optional[torch.Tensor]],
            Tuple[torch.Tensor, List[torch.Tensor]],
        ]
        self._init_forward()

        self._update_coefficient_config()

        self._coefficient: ProbabilityParameter[T, TC]
        self.init_coefficient()

    id = ReadOnlyDescriptor[int]()
    block_dim = ReadOnlyDescriptor[int]()
    block_state_binding = ReadOnlyDescriptor[bool]()

    def bind_width(self, layer: "DualTransitionLayer") -> None:
        if layer.block_dim != self._block_dim:
            raise ValueError(
                f"Cannot bind layer with different block_dim. "
                f"Current block_dim is {self._block_dim} while the given layer has block_dim {layer.block_dim}."
            )
        for b, block in enumerate(self._blocks):
            cast(
                BaseDualTransitionBlock[T, TC], block
            ).matrix_parameter.bind_with(
                cast(
                    BaseDualTransitionBlock[T, TC], layer.blocks[b]
                ).matrix_parameter
            )

    def _init_forward(self) -> None:
        if self._block_state_binding:

            for block in self._blocks:
                cast(
                    BaseDualTransitionBlock[T, TC], block
                ).initial_state_parameter.bind_with(self._initial_state)
            self._forward = self._forward_bound_initial_state
        else:
            del self._initial_state
            self._forward = self._forward_unbound_initial_state

    def _update_coefficient_config(self) -> None:
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

    def init_coefficient(self) -> None:
        self._coefficient = ProbabilityParameter[T, TC](
            self._config.coefficient.probability_parameter,
            (
                (self._state_dim, self._block_dim)
                if self._block_state_binding
                else (self._block_dim,)
            ),
        )

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

    def delete_coefficient(self) -> None:
        del self._coefficient

    @property
    def blocks(self) -> List[BaseDualTransitionBlock[T, TC]]:
        return [
            cast(BaseDualTransitionBlock[T, TC], block)
            for block in self._blocks
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
                cast(BaseDualTransitionBlock[T, TC], block).matrix
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

    def _set_internal_estimated_state(
        self, estimated_state: torch.Tensor, predicted_next_state: torch.Tensor
    ) -> None:
        if self._inference:
            super()._set_internal_estimated_state(
                estimated_state, predicted_next_state
            )

    def _forward_bound_initial_state(
        self,
        likelihood_trajectory: torch.Tensor,
        coefficient: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        estimated_state_trajectory, control_trajectory = self.process(
            likelihood_trajectory
        )

        return estimated_state_trajectory, control_trajectory

    def _forward_unbound_initial_state(
        self,
        likelihood_trajectory: torch.Tensor,
        coefficient: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # TODO: the implementation of the unbound initial state is not correct.

        if coefficient is None:
            coefficient = self.coefficient.unsqueeze(dim=0).unsqueeze(
                dim=0
            )  # (1, 1, block_dim,)

        estimated_state_trajectory = torch.zeros_like(
            likelihood_trajectory,
            device=likelihood_trajectory.device,
        )  # (batch_size, horizon, state_dim)
        # predicted_next_state_trajectory = torch.zeros_like(
        #     likelihood_state_trajectory,
        #     device=likelihood_state_trajectory.device,
        # )  # (batch_size, horizon, state_dim)

        for b, block in enumerate(self._blocks):
            _estimated_state_trajectory = block(
                likelihood_trajectory
            )  # (batch_size, horizon, state_dim), (batch_size, horizon, state_dim)

            estimated_state_trajectory = weighted_sum(
                estimated_state_trajectory,
                _estimated_state_trajectory,
                coefficient[:, :, b],
            )
            # predicted_next_state_trajectory = weighted_sum(
            #     predicted_next_state_trajectory,
            #     _predicted_next_state_trajectory,
            #     coefficient[:, :, b],
            # )

        control_trajectory = torch.zeros_like(
            estimated_state_trajectory,
            device=estimated_state_trajectory.device,
        )

        return estimated_state_trajectory, [control_trajectory]

    def forward(
        self,
        likelihood_trajectory: torch.Tensor,
        coefficient: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        (estimated_state_trajectory, control_trajectories) = self._forward(
            likelihood_trajectory,
            coefficient,
        )

        # for control_trajectory in control_trajectories:
        #     logger.info(
        #         control_trajectory.shape
        #     )
        # quit()
        return estimated_state_trajectory, control_trajectories

    def reset(self) -> None:
        for block in self._blocks:
            reset_module(block)
        if self._block_state_binding:
            super().reset()
