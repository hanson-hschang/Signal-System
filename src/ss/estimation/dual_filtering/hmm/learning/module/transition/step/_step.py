from typing import Any, Generic, List, Tuple, cast

import torch
import torch.nn as nn

from ss.estimation.dual_filtering.hmm.learning.module.filter.config import (
    DualFilterConfig,
)
from ss.estimation.dual_filtering.hmm.learning.module.transition.step.config import (
    DualTransitionStepConfig,
)
from ss.utility.learning.parameter.probability import ProbabilityParameter
from ss.utility.learning.parameter.transformer import T
from ss.utility.learning.parameter.transformer.config import TC


class DualTransitionStepMixin(nn.Module, Generic[T, TC]):
    def __init__(
        self,
        step_config: DualTransitionStepConfig[TC],
        filter_config: DualFilterConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._state_dim = filter_config.state_dim
        self._skip_first = step_config.skip_first

        self._initial_state: ProbabilityParameter[
            T, TC
        ] = ProbabilityParameter[T, TC](
            step_config.initial_state.probability_parameter,
            (self._state_dim,),
        )

        self._terminal_dual_function = torch.eye(
            self._state_dim,
            device=self._initial_state.pytorch_parameter.device,
        )

        self._is_initialized = False
        self._estimated_state = (
            torch.ones(self._state_dim) / self._state_dim
        ).repeat(
            1, 1
        )  # (batch_size, state_dim)
        self._matrix: ProbabilityParameter[T, TC]

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
        raise NotImplementedError

    @matrix.setter
    def matrix(self, matrix: torch.Tensor) -> None:
        raise NotImplementedError

    @property
    def estimated_state(self) -> torch.Tensor:
        batch_size, _ = self._estimated_state.shape
        if batch_size == 1:
            return self._estimated_state.squeeze(0)
        return self._estimated_state

    # @property
    # def predicted_next_state(self) -> torch.Tensor:
    #     transition_matrix = self.matrix  # (state_dim, state_dim)
    #     return self._prediction(self.estimated_state, transition_matrix)

    def reset(self) -> None:
        self._is_initialized = False
        self._get_internal_estimated_state()
        self._is_initialized = False

    # @torch.compile
    # def _prediction(
    #     self,
    #     estimated_state: torch.Tensor,
    #     transition_matrix: torch.Tensor,
    # ) -> torch.Tensor:
    #     predicted_next_state = torch.matmul(estimated_state, transition_matrix)
    #     return predicted_next_state

    # @torch.compile
    # def _update(
    #     self,
    #     prior_state: torch.Tensor,
    #     likelihood_state: torch.Tensor,
    # ) -> torch.Tensor:
    #     # update step based on likelihood_state (conditional probability)
    #     posterior_state = nn.functional.normalize(
    #         prior_state * likelihood_state,
    #         p=1,
    #         dim=1,
    #     )  # (batch_size, state_dim)
    #     return posterior_state

    # @torch.compile
    # def _first_prediction(
    #     self,
    #     estimated_state: torch.Tensor,
    #     transition_matrix: torch.Tensor,
    # ) -> torch.Tensor:
    #     predicted_next_state = (
    #         estimated_state
    #         if self._skip_first
    #         else self._prediction(estimated_state, transition_matrix)
    #     )  # (batch_size, state_dim)
    #     return predicted_next_state

    def _get_internal_estimated_state(
        self, batch_size: int = 1
    ) -> torch.Tensor:
        if not self._is_initialized:
            self._is_initialized = True
            self._estimated_state = self.initial_state.repeat(batch_size, 1)
        return self._estimated_state

    def _set_internal_estimated_state(
        self, estimated_state: torch.Tensor, predicted_next_state: torch.Tensor
    ) -> None:
        self._estimated_state = (
            predicted_next_state if self._skip_first else estimated_state
        )

    # def process(
    #     self,
    #     likelihood_state_trajectory: torch.Tensor,
    #     emission_difference_trajectory: torch.Tensor,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     batch_size, horizon, _ = likelihood_state_trajectory.shape
    #     # (batch_size, horizon, state_dim)

    #     transition_matrix = self.matrix  # (state_dim, state_dim)

    #     estimated_state_trajectory = torch.empty(
    #         (batch_size, horizon, self._state_dim),
    #         device=likelihood_state_trajectory.device,
    #     )

    #     predicted_next_state_trajectory = torch.empty(
    #         (batch_size, horizon, self._state_dim),
    #         device=likelihood_state_trajectory.device,
    #     )

    #     dual_function_history = torch.empty(
    #         (
    #             batch_size,
    #             horizon + 1,
    #             self._state_dim,
    #             self._state_dim,
    #         ),
    #         device=likelihood_state_trajectory.device,
    #     )
    #     dual_function_history[:, -1, :, :] = (
    #         self._terminal_dual_function.repeat(batch_size, 1, 1)
    #     )

    #     control_history = torch.empty(
    #         (batch_size, horizon, self._state_dim),
    #         device=likelihood_state_trajectory.device,
    #     )

    #     initial_state = self._get_internal_estimated_state(
    #         batch_size
    #     )  # (batch_size, state_dim)

    #     for k in range(horizon):
    #         dual_function = dual_function_history[:, -1 - k, :, :]
    #         emission_difference = emission_difference_trajectory[:, -1 - k, :]
    #         estimated_distribution = (
    #             initial_state
    #             if k == (horizon - 1)
    #             else likelihood_state_trajectory[:, -1 - k - 1, :]
    #         )

    #         control = self._compute_control(
    #             self._compute_one_transition(transition_matrix, dual_function),
    #             emission_difference,
    #             estimated_distribution,
    #         )

    #         dual_function_history[:, -1 - k - 1, :, :] = self._backward_step(
    #             transition_matrix,
    #             dual_function,
    #             emission_difference,
    #             control,
    #         )

    #         control_history[:, -1 - k, :] = control

    #     estimator = torch.empty(
    #         (batch_size, horizon + 1, self._state_dim),
    #         device=likelihood_state_trajectory.device,
    #     )

    #     estimator[:, 0, :] = self._batch_vector_matrix_mul(
    #         initial_state, dual_function_history[:, 0, :, :]
    #     )

    #     for k in range(horizon):
    #         estimator[:, k + 1, :] = (
    #             estimator[:, k, :] - control_history[:, k, :]
    #         )
    #         predicted_next_state_trajectory[:, k, :] = (
    #             self._update_estimated_distribution(
    #                 dual_function_history[:, k + 1, :, :],
    #                 estimator[:, k + 1, :],
    #             )
    #         )
    #         # predicted_next_state_trajectory[:, k, :] = (estimator[:, k + 1, :].clone() / torch.sum(
    #         #     estimator[:, k + 1, :], dim=1, keepdim=True
    #         # ))

    #     return estimated_state_trajectory, predicted_next_state_trajectory

    # method 2
    def process(
        self,
        likelihood_state_trajectory: torch.Tensor,
        # emission_difference_trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        batch_size, horizon, _ = likelihood_state_trajectory.shape
        # (batch_size, horizon, state_dim)

        initial_state = self._get_internal_estimated_state(
            batch_size
        )  # (batch_size, state_dim)
        transition_matrix = self.matrix  # (state_dim, state_dim)

        estimated_state_trajectory = torch.empty(
            (batch_size, horizon + 1, self._state_dim),
            device=likelihood_state_trajectory.device,
        )
        estimated_state_trajectory[:, 0, :] = initial_state

        emission_difference_trajectory = likelihood_state_trajectory * 2 - 1

        # predicted_next_state_trajectory = torch.empty(
        #     (batch_size, horizon, self._state_dim),
        #     device=emission_difference_trajectory.device,
        # )

        control_trajectories = []

        for k in range(horizon):

            (
                dual_function_trajectory,  # (batch_size, horizon, state_dim, number_of_dual_function)
                control_trajectory,  # (batch_size, horizon, number_of_dual_function)
            ) = self._compute_backward_path(
                emission_difference_trajectory[:, : k + 1, :],
                estimated_state_trajectory[:, : k + 1, :],
                transition_matrix,
            )

            estimated_state_trajectory[:, k + 1, :] = (
                self._compute_terminal_estimated_state(
                    initial_state,
                    dual_function_trajectory[:, 0, :, :],
                    control_trajectory,
                )
            )

            control_trajectories.append(control_trajectory)

            # dual_function_history = torch.empty(
            #     (
            #         batch_size,
            #         _horizon + 1,
            #         self._state_dim,
            #         self._state_dim,
            #     ),
            #     device=emission_difference_trajectory.device,
            # )
            # dual_function_history[:, -1, :, :] = (
            #     self._terminal_dual_function.repeat(batch_size, 1, 1)
            # )

            # control_history = torch.empty(
            #     (batch_size, _horizon, self._state_dim),
            #     device=emission_difference_trajectory.device,
            # )

            # initial_state = self._get_internal_estimated_state(
            #     batch_size
            # )  # (batch_size, state_dim)

            # for k in range(_horizon + 1):
            #     dual_function = dual_function_history[:, -1 - k, :, :]
            #     emission_difference = emission_difference_trajectory[
            #         :, _horizon + 1 - k, :
            #     ]
            #     estimated_distribution = (
            #         initial_state
            #         if k == (_horizon - 1)
            #         else likelihood_state_trajectory[
            #             :, _horizon - 1 - k - 1, :
            #         ]
            #     )

            #     control = self._compute_control(
            #         self._compute_one_transition(
            #             transition_matrix, dual_function
            #         ),
            #         emission_difference,
            #         estimated_distribution,
            #     )

            #     dual_function_history[:, -1 - k - 1, :, :] = (
            #         self._backward_step(
            #             transition_matrix,
            #             dual_function,
            #             emission_difference,
            #             control,
            #         )
            #     )

            #     control_history[:, -1 - k, :] = control

            # estimator = torch.empty(
            #     (batch_size, _horizon + 1, self._state_dim),
            #     device=likelihood_state_trajectory.device,
            # )

            # estimator[:, 0, :] = self._batch_vector_matrix_mul(
            #     initial_state, dual_function_history[:, 0, :, :]
            # )

            # for k in range(_horizon):
            #     estimator[:, k + 1, :] = (
            #         estimator[:, k, :] - control_history[:, k, :]
            #     )
            #     # predicted_next_state_trajectory[:, k, :] = (
            #     #     self._update_estimated_distribution(
            #     #         dual_function_history[:, k + 1, :, :],
            #     #         estimator[:, k + 1, :],
            #     #     )
            #     # )
            # _estimator = torch.max(
            #     estimator[:, -1, :], torch.tensor(1e-6).to(estimator.device)
            # )

            # predicted_next_state_trajectory[:, _horizon - 1, :] = (
            #     _estimator.clone() / torch.sum(_estimator, dim=1, keepdim=True)
            # )

        return estimated_state_trajectory[:, 1:, :], control_trajectories

    def _compute_backward_path(
        self,
        emission_difference_trajectory: torch.Tensor,
        estimated_distribution_trajectory: torch.Tensor,
        transition_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, horizon, state_dim = emission_difference_trajectory.shape

        dual_function_trajectory = torch.empty(
            (
                batch_size,
                horizon + 1,
                state_dim,
                state_dim,
            ),
            device=emission_difference_trajectory.device,
        )

        dual_function_trajectory[:, -1, :, :] = (
            self._terminal_dual_function.repeat(batch_size, 1, 1)
        )

        control_trajectory = torch.empty(
            (batch_size, horizon, state_dim),
            device=emission_difference_trajectory.device,
        )

        # implementation of the backward path
        for k in range(horizon, 0, -1):
            dual_function = dual_function_trajectory[:, k, :, :]
            emission_difference = emission_difference_trajectory[
                :, k - 1, :
            ]  # (batch_size, state_dim)
            past_estimated_distribution = estimated_distribution_trajectory[
                :, k - 1, :
            ]

            past_dual_function = self._compute_one_transition(
                transition_matrix, dual_function
            )

            # Compute the control
            control = self._compute_control(
                past_dual_function,
                emission_difference,
                past_estimated_distribution,
            )

            dual_function_trajectory[:, k - 1, :, :] = self._backward_step(
                past_dual_function,
                emission_difference,
                control,
            )

            control_trajectory[:, k - 1, :] = control

        return dual_function_trajectory, control_trajectory

    def _compute_terminal_estimated_state(
        self,
        initial_state: torch.Tensor,
        dual_function: torch.Tensor,
        control_trajectory: torch.Tensor,
    ) -> torch.Tensor:

        terminal_estimated_state = self._batch_vector_matrix_mul(
            initial_state, dual_function
        )  # (batch_size, number_of_dual_function)

        terminal_estimated_state -= torch.sum(
            control_trajectory,
            dim=1,
        )

        terminal_estimated_state = torch.clamp(
            terminal_estimated_state,
            min=0,
        )

        terminal_estimated_state = terminal_estimated_state / torch.sum(
            terminal_estimated_state,
            dim=1,
            keepdim=True,
        )

        return terminal_estimated_state

    def _compute_one_transition(
        self,
        row_vectors: torch.Tensor,
        batch_tensor: torch.Tensor,
    ) -> torch.Tensor:
        # transition_matrix: (state_dim, state_dim)
        # batch_tensor: (batch_size, state_dim, dual_function_dim)

        result = torch.einsum(
            "nm, bmd ->bnd", row_vectors, batch_tensor
        )  # (batch_size, state_dim, dual_function_dim)
        return result

    def _compute_control(
        self,
        past_dual_function: torch.Tensor,
        emission_difference: torch.Tensor,
        estimated_distribution: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, dual_function_dim = past_dual_function.shape
        # control = torch.empty(
        #     (batch_size, dual_function_dim),
        #     device=dual_function.device,
        # )

        expected_emission = torch.einsum(
            "bn,bn->b",
            estimated_distribution.clone(),
            emission_difference.clone(),
        )  # (batch_size,)
        denominator = -(expected_emission**2) + 1  # (batch_size,)

        control = (
            torch.einsum(
                "bn, bnd -> bd",
                estimated_distribution.clone(),
                past_dual_function.clone(),
            )
            * expected_emission.unsqueeze(1).clone()
        ) - torch.einsum(
            "bn, bnd -> bd",
            estimated_distribution.clone(),
            past_dual_function * emission_difference.unsqueeze(2).clone(),
        )

        # control = torch.zeros(
        #     (batch_size, dual_function_dim),
        #     device=emission_difference.device,
        # )

        # mean_zero_emission_difference = (
        #     emission_difference - expected_emission.unsqueeze(1)
        # )  # (batch_size, state_dim)

        # temp = torch.einsum(
        #     "bnd,bn->bnd", dual_function, mean_zero_emission_difference
        # )  # (batch_size, state_dim, dual_function_dim)
        # control = -torch.einsum(
        #     "bi,bid->bd", estimated_distribution, temp
        # )  # (batch_size, dual_function_dim)

        zero_mask = denominator == 0
        nonzero_mask = torch.logical_not(zero_mask)
        control = control.clone()  # (batch_size, dual_function_dim)
        if zero_mask.any():
            control[zero_mask] = control[zero_mask] * 0
            control[nonzero_mask] = (
                control[nonzero_mask]
                / denominator[nonzero_mask].unsqueeze(1).clone()
            )
        else:
            # control[zero_mask] = control[zero_mask] * 0
            # denominator[zero_mask] = denominator[zero_mask] + 1.0
            # control[not zero_mask] = control[not zero_mask] / denominator[
            #     not zero_mask
            # ].unsqueeze(1)
            # else:
            control = control / denominator.unsqueeze(1)

        return control

    def _backward_step(
        self,
        past_dual_function: torch.Tensor,
        emission_difference: torch.Tensor,
        control: torch.Tensor,
    ) -> torch.Tensor:
        result = past_dual_function + self._batch_outer(
            emission_difference, control
        )
        return result

    def _batch_outer(
        self,
        batch_vector_1: torch.Tensor,
        batch_vector_2: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.einsum("bi, bj->bij", batch_vector_1, batch_vector_2)
        return result

    def _batch_vector_matrix_mul(
        self,
        initial_distribution: torch.Tensor,
        dual_function: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.einsum(
            "bi, bij ->bj", initial_distribution, dual_function
        )
        return result

    def _update_estimated_distribution(
        self,
        dual_function: torch.Tensor,
        estimator: torch.Tensor,
    ) -> torch.Tensor:
        result = cast(
            torch.Tensor,
            torch.linalg.lstsq(
                torch.transpose(dual_function, 1, 2).clone(),
                estimator.unsqueeze(2).clone(),
            ).solution,
        ).squeeze(2)

        # Find negative values and create a mask
        negative_mask = result < 0
        if negative_mask.any():
            # print(result[negative_mask])
            # input(
            #     "Negative values found in the result tensor. Press Enter to continue."
            # )
            # Set negative values to zero
            result[negative_mask] = result[negative_mask] * 0
            # result[negative_mask] = result[negative_mask] + 1e-6

        result = result / torch.sum(result, dim=1, keepdim=True)

        return result
