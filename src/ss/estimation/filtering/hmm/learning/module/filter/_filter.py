from typing import Callable, Generic, assert_never

import torch

from ss.estimation.filtering.hmm.learning.module.filter.config import (
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
        self._predicted_state: torch.Tensor
        self._init_state()

    state_dim = ReadOnlyDescriptor[int]()
    discrete_observation_dim = ReadOnlyDescriptor[int]()
    estimation_dim = ReadOnlyDescriptor[int]()
    batch_size = BatchSizeDescriptor()

    def _init_state(self) -> None:
        self._estimated_state = torch.zeros(self._batch_size, self._state_dim)
        self._predicted_state = torch.zeros(self._batch_size, self._state_dim)

    estimated_state = BatchTensorDescriptor("_batch_size", "_state_dim")
    predicted_state = BatchTensorDescriptor("_batch_size", "_state_dim")
