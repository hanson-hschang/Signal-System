from typing import Generic, Optional, Tuple, cast

import torch
from torch import nn

from ss.estimation.filtering.hmm.learning.module.filter.config import (
    DualFilterConfig,
)
from ss.estimation.filtering.hmm.learning.module.transition.config import (
    TransitionConfig,
)
from ss.utility.descriptor import BatchTensorDescriptor
from ss.utility.learning.module import BaseLearningModule, reset_module
from ss.utility.learning.parameter.probability import ProbabilityParameter
from ss.utility.learning.parameter.transformer import T
from ss.utility.learning.parameter.transformer.config import TC
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class DualTransitionModule(
    BaseLearningModule[TransitionConfig[TC]], Generic[T, TC]
):
    def __init__(
        self,
        config: TransitionConfig[TC],
        filter_config: DualFilterConfig,
    ) -> None:
        super().__init__(config)
        self._state_dim = filter_config.state_dim
        self._initial_state: ProbabilityParameter[
            T, TC
        ] = ProbabilityParameter[T, TC](
            self._config.initial_state.probability_parameter,
            (self._state_dim,),
        )
        self._matrix = ProbabilityParameter[T, TC](
            self._config.matrix.probability_parameter,
            (self._state_dim, self._state_dim),
        )
        self._terminal_dual_function = torch.eye(
            self._state_dim,
            device=self._initial_state.pytorch_parameter.device,
        )

        self._init_batch_size(batch_size=1)

    estimated_state = BatchTensorDescriptor(
        "_batch_size",
        "_state_dim",
    )

    def _init_batch_size(
        self, batch_size: int, is_initialized: bool = False
    ) -> None:
        self._is_initialized = is_initialized
        self._batch_size = batch_size
        with self.evaluation_mode():
            self._estimated_state = self.initial_state.repeat(
                self._batch_size, 1
            )

    def _check_batch_size(self, batch_size: int) -> None:
        if self._is_initialized:
            assert batch_size == self._batch_size, (
                f"batch_size must be the same as the initialized batch_size. "
                f"batch_size given is {batch_size} while the initialized batch_size is {self._batch_size}."
            )
            return
        self._init_batch_size(batch_size, is_initialized=True)

    @property
    def initial_state_parameter(
        self,
    ) -> ProbabilityParameter[T, TC]:
        return self._initial_state

    @property
    def initial_state(self) -> torch.Tensor:
        initial_state: torch.Tensor = self._initial_state()
        return initial_state

    @initial_state.setter
    def initial_state(self, initial_state: torch.Tensor) -> None:
        self._initial_state.set_value(initial_state)

    @property
    def matrix_parameter(
        self,
    ) -> ProbabilityParameter[T, TC]:
        return self._matrix

    @property
    def matrix(self) -> torch.Tensor:
        matrix: torch.Tensor = self._matrix()
        return matrix

    @matrix.setter
    def matrix(self, matrix: torch.Tensor) -> None:
        self._matrix.set_value(matrix)

    def _compute_past_dual_function(
        self,
        transition_matrix: torch.Tensor,
        dual_function: torch.Tensor,
    ) -> torch.Tensor:
        # transition_matrix: (state_dim, state_dim)
        # dual_function: (batch_size, state_dim, state_dim)

        past_dual_function = torch.einsum(
            "ij, bjk -> bik", transition_matrix, dual_function
        )
        return past_dual_function

    def _compute_control(
        self,
        dual_function: torch.Tensor,
        emission_difference: torch.Tensor,
        estimated_state_distribution: torch.Tensor,
    ) -> torch.Tensor:
        # dual_function: (batch_size, state_dim, number_of_dual_functions)
        # emission_difference: (batch_size, state_dim)
        # estimated_state_distribution: (batch_size, state_dim)
        batch_size, _, number_of_dual_functions = dual_function.shape

        control = torch.empty(
            (batch_size, number_of_dual_functions),
            device=dual_function.device,
            dtype=dual_function.dtype,
        )  # (batch_size, number_of_dual_functions)

        expected_emission = torch.einsum(
            "bi, bi -> b", estimated_state_distribution, emission_difference
        )  # (batch_size,)
        denominator = 1 - (expected_emission**2)  # (batch_size,)

        nonzero_indices = denominator != 0
        mean_zero_emission_difference = emission_difference[
            nonzero_indices, :
        ] - expected_emission[nonzero_indices].unsqueeze(
            1
        )  # (batch_size, state_dim)
        new_measure = (
            estimated_state_distribution[nonzero_indices, :]
            * mean_zero_emission_difference
        )  # (batch_size, state_dim)
        control[nonzero_indices, :] = -torch.einsum(
            "bi, bij -> bj", new_measure, dual_function[nonzero_indices, :, :]
        ) / denominator[nonzero_indices].unsqueeze(
            1
        )  # (batch_size, number_of_dual_functions)
        control[~nonzero_indices, :] = 0.0

        return control

    def _backward_dual_function_step(
        self,
        past_dual_function: torch.Tensor,
        emission_difference: torch.Tensor,
        control: torch.Tensor,
    ) -> torch.Tensor:
        # past_dual_function: (batch_size, state_dim, number_of_dual_functions)
        # emission_difference: (batch_size, state_dim)
        # control: (batch_size, number_of_dual_functions)
        updated_past_dual_function = past_dual_function + torch.einsum(
            "bi, bj -> bij", emission_difference, control
        )
        return updated_past_dual_function

    def _compute_backward_path(
        self,
        transition_matrix: torch.Tensor,
        terminal_dual_function: torch.Tensor,
        emission_difference_trajectory: torch.Tensor,
        estimated_state_distribution_trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # transition_matrix: (state_dim, state_dim)
        # terminal_dual_function: (state_dim, state_dim)
        # emission_difference_trajectory: (batch_size, state_dim, horizon)
        # estimated_state_distribution_trajectory: (batch_size, state_dim, horizon)

        batch_size, state_dim, horizon = emission_difference_trajectory.shape

        dual_function_history = torch.empty(
            (batch_size, state_dim, state_dim, horizon + 1),
            device=emission_difference_trajectory.device,
            dtype=emission_difference_trajectory.dtype,
        )
        dual_function_history[:, :, :, -1] = terminal_dual_function.repeat(
            batch_size, 1, 1
        )
        control_history = torch.empty(
            (batch_size, state_dim, horizon),
            device=emission_difference_trajectory.device,
            dtype=emission_difference_trajectory.dtype,
        )

        for k in range(horizon, 0, -1):
            dual_function = self._compute_past_dual_function(
                transition_matrix,
                dual_function_history[:, :, :, k],
            )
            emission_difference = emission_difference_trajectory[:, :, k - 1]

            control = self._compute_control(
                dual_function,
                emission_difference,
                estimated_state_distribution_trajectory[:, :, k - 1],
            )

            dual_function_history[:, :, :, k - 1] = (
                self._backward_dual_function_step(
                    dual_function,
                    emission_difference,
                    control,
                )
            )
            control_history[:, :, k - 1] = control.clone()

        return dual_function_history, control_history

    def _compute_estimated_state_distribution(
        self,
        initial_estimated_state_distribution: torch.Tensor,
        initial_dual_function: torch.Tensor,
        control_history: torch.Tensor,
    ) -> torch.Tensor:
        # initial_estimated_state_distribution: (batch_size, state_dim)
        # initial_dual_function: (batch_size, state_dim, number_of_dual_functions)
        # control_history: (batch_size, number_of_dual_functions, horizon)
        estimated_state_distribution = torch.einsum(
            "bi, bij -> bj",
            initial_estimated_state_distribution,
            initial_dual_function,
        ) - torch.sum(control_history, dim=2)
        return estimated_state_distribution

    def _normalize_distribution(
        self, distribution: torch.Tensor
    ) -> torch.Tensor:
        # distribution: (batch_size, state_dim)
        positive_indices = distribution > 0.0
        clamped_distribution = torch.empty_like(distribution)
        clamped_distribution[positive_indices] = distribution[positive_indices]
        clamped_distribution[~positive_indices] = (
            0.0 * distribution[~positive_indices]
        )
        # clamped_distribution = torch.clamp(distribution, min=0.0)
        normalized_distribution = nn.functional.normalize(
            clamped_distribution,
            p=1,
            dim=1,
        )
        return normalized_distribution

    def _process(
        self,
        estimated_state_distribution_trajectory: torch.Tensor,
        emission_difference_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        # estimated_state_trajectory: (batch_size, state_dim, horizon)
        # emission_difference_trajectory: (batch_size, state_dim, horizon)

        dual_function_history, control_history = self._compute_backward_path(
            self.matrix,  # (state_dim, state_dim)
            self._terminal_dual_function,
            emission_difference_trajectory,
            estimated_state_distribution_trajectory,
        )  # (batch_size, state_dim, state_dim, horizon+1), (batch_size, state_dim, horizon)

        estimated_state_distribution = (
            self._compute_estimated_state_distribution(
                estimated_state_distribution_trajectory[:, :, 0],
                dual_function_history[:, :, :, 0],
                control_history,
            )
        )  # (batch_size, state_dim)

        return self._normalize_distribution(estimated_state_distribution)

    def forward(
        self,
        emission_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, horizon = emission_trajectory.shape

        estimated_state_distribution_trajectory = torch.empty(
            (batch_size, self._state_dim, horizon + 1),
            device=emission_trajectory.device,
            dtype=emission_trajectory.dtype,
        )

        estimated_state_distribution_trajectory[:, :, 0] = (
            self.initial_state.repeat(batch_size, 1)
        )

        emission_difference_trajectory = 2 * emission_trajectory - 1

        for k in range(1, horizon + 1):

            estimated_state = self._process(
                estimated_state_distribution_trajectory[:, :, :k].clone(),
                emission_difference_trajectory[:, :, :k],
            )

            estimated_state_distribution_trajectory[:, :, k] = (
                estimated_state.clone()
            )

        return estimated_state_distribution_trajectory[:, :, 1:]

    @torch.inference_mode()
    def at_inference(
        self,
        emission_difference_trajectory: torch.Tensor,
        estimated_state_distribution_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        self._estimated_state = self._process(
            estimated_state_distribution_trajectory,
            emission_difference_trajectory,
        )
        return self.estimated_state

    def reset(self, batch_size: Optional[int] = None) -> None:
        self._init_batch_size(
            batch_size=self._batch_size if batch_size is None else batch_size,
            is_initialized=False,
        )
