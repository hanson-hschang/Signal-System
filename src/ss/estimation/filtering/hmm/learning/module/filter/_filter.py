from typing import Callable, Generic, assert_never

import torch

from ss.estimation.filtering.hmm.learning.module.filter.config import (
    DualFilterConfig,
    FilterConfig,
)
from ss.utility.assertion.validator import PositiveIntegerValidator
from ss.utility.descriptor import (
    BatchTensorDescriptor,
    Descriptor,
    ReadOnlyDescriptor,
)
from ss.utility.learning.module import BaseLearningModule


class FilterModule(
    BaseLearningModule[FilterConfig],
):
    class BatchSizeDescriptor(Descriptor[int, "FilterModule"]):
        def __set__(self, instance: "FilterModule", value: int) -> None:
            value = PositiveIntegerValidator(value).get_value()
            super().__set__(instance, value)
            instance._init_state()

    def __init__(
        self,
        config: FilterConfig,
    ) -> None:
        super().__init__(config)
        self._state_dim = self._config.state_dim
        self._discrete_observation_dim = self._config.discrete_observation_dim
        self._estimation_dim = self._config.estimation_dim

        self._batch_size = 1
        self._estimated_state: torch.Tensor
        # self._predicted_state: torch.Tensor
        self._init_state()

    state_dim = ReadOnlyDescriptor[int]()
    discrete_observation_dim = ReadOnlyDescriptor[int]()
    estimation_dim = ReadOnlyDescriptor[int]()
    batch_size = BatchSizeDescriptor()

    def _init_state(self) -> None:
        self._estimated_state = torch.zeros(self._batch_size, self._state_dim)
        # self._predicted_state = torch.zeros(self._batch_size, self._state_dim)

    estimated_state = BatchTensorDescriptor("_batch_size", "_state_dim")
    # predicted_state = BatchTensorDescriptor("_batch_size", "_state_dim")


class DualFilterModule(
    BaseLearningModule[DualFilterConfig],
):
    class BatchSizeDescriptor(Descriptor[int, "DualFilterModule"]):
        def __set__(self, instance: "DualFilterModule", value: int) -> None:
            value = PositiveIntegerValidator(value).get_value()
            super().__set__(instance, value)
            instance._init_state()

    def __init__(
        self,
        config: DualFilterConfig,
    ) -> None:
        super().__init__(config)
        self._state_dim = self._config.state_dim
        self._dual_function_dim = self._config.state_dim
        self._observation_dim = 1
        self._discrete_observation_dim = self._config.discrete_observation_dim
        self._estimation_dim = self._config.estimation_dim
        self._history_horizon = self._config.history_horizon
        self._state_history_horizon = self._history_horizon + 1

        self._batch_size = 1
        self._init_state()

    state_dim = ReadOnlyDescriptor[int]()
    # control_dim = ReadOnlyDescriptor[int]()
    observation_dim = ReadOnlyDescriptor[int]()
    discrete_observation_dim = ReadOnlyDescriptor[int]()
    estimation_dim = ReadOnlyDescriptor[int]()
    # current_history_horizon = ReadOnlyDescriptor[int]()
    history_horizon = ReadOnlyDescriptor[int]()
    batch_size = BatchSizeDescriptor()

    def _init_state(self) -> None:
        self._current_history_horizon = 0
        self._estimated_state = torch.full(
            (self._batch_size, self._state_dim), float("nan")
        )
        self._estimated_state_history = torch.full(
            (self._batch_size, self._state_dim, self._state_history_horizon),
            float("nan"),
        )
        # self._observation_history = torch.zeros(
        #     self._batch_size, self._observation_dim, self._history_horizon
        # )
        self._emission_difference_history = torch.full(
            (self._batch_size, self._state_dim, self._history_horizon),
            float("nan"),
        )
        self._control_history = torch.full(
            (self._batch_size, self._dual_function_dim, self._history_horizon),
            float("nan"),
        )

    estimated_state = BatchTensorDescriptor(
        "_batch_size",
        "_state_dim",
    )
    estimated_state_history = BatchTensorDescriptor(
        "_batch_size",
        "_state_dim",
        "_state_history_horizon",
    )
    emission_difference_history = BatchTensorDescriptor(
        "_batch_size",
        "_state_dim",
        "_history_horizon",
    )
    # observation_history = BatchTensorDescriptor(
    #     "_batch_size",
    #     "_observation_dim",
    #     "_history_horizon",
    # )
    control_history = BatchTensorDescriptor(
        "_batch_size",
        "_dual_function_dim",
        "_history_horizon",
    )

    @property
    def emission_difference_trajectory(self) -> torch.Tensor:
        return torch.flip(
            self._emission_difference_history[
                :, :, : self._current_history_horizon
            ],
            dims=(2,),
        )

    @property
    def estimated_state_trajectory(self) -> torch.Tensor:
        return torch.flip(
            self._estimated_state_history[
                :, :, : self._current_history_horizon
            ],
            dims=(2,),
        )

    @torch.inference_mode()
    def update_emission(self, emission: torch.Tensor) -> None:
        self._emission_difference_history[:, :, -1] = 2 * emission - 1
        self._emission_difference_history = torch.roll(
            self._emission_difference_history, 1, dims=2
        )
        self._current_history_horizon = min(
            self._current_history_horizon + 1, self._history_horizon
        )

    @torch.inference_mode()
    def update_state(self, estimated_state: torch.Tensor) -> None:
        self._estimated_state_history[:, :, -1] = estimated_state
        self._estimated_state_history = torch.roll(
            self._estimated_state_history, 1, dims=2
        )
        self._estimated_state = estimated_state

    def reset(  # type: ignore
        self, initial_state_distribution: torch.Tensor, batch_size: int = 1
    ) -> None:
        self._batch_size = PositiveIntegerValidator(batch_size).get_value()
        self._init_state()
        self._estimated_state = initial_state_distribution.repeat(
            self._batch_size, 1
        )
        self._estimated_state_history[:, :, 0] = self._estimated_state
