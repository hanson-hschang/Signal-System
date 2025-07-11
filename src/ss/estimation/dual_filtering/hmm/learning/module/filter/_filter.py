from typing import Callable, Generic, assert_never

import torch

from ss.estimation.dual_filtering.hmm.learning.module.filter.config import (
    DualFilterConfig,
)
from ss.utility.assertion.validator import PositiveIntegerValidator
from ss.utility.descriptor import (
    BatchTensorDescriptor,
    Descriptor,
    ReadOnlyDescriptor,
)
from ss.utility.learning.module import BaseLearningModule
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


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
        self._discrete_observation_dim = self._config.discrete_observation_dim
        self._estimation_dim = self._config.estimation_dim
        self._history_length = 0
        self._max_history_length = self._config.history_length

        self._batch_size = 1
        self._estimated_state: torch.Tensor
        # self._predicted_state: torch.Tensor
        self._init_state()

    state_dim = ReadOnlyDescriptor[int]()
    discrete_observation_dim = ReadOnlyDescriptor[int]()
    estimation_dim = ReadOnlyDescriptor[int]()
    history_length = ReadOnlyDescriptor[int]()
    max_history_length = ReadOnlyDescriptor[int]()
    batch_size = BatchSizeDescriptor()

    def _init_state(self) -> None:
        self._estimated_state = torch.zeros(self._batch_size, self._state_dim)
        # self._predicted_state = torch.zeros(self._batch_size, self._state_dim)
        self._estimated_state_history = torch.zeros(
            self._batch_size, self._max_history_length, self._state_dim
        )
        self._emission_history = torch.zeros(
            self._batch_size, self._max_history_length, self._state_dim
        )
        self._control_history = torch.zeros(
            self._batch_size, self._max_history_length, self._state_dim
        )
        # self._emission_difference_history = torch.zeros(
        #     self._batch_size, self._history_length, self._state_dim
        # )

    estimated_state = BatchTensorDescriptor("_batch_size", "_state_dim")
    # predicted_state = BatchTensorDescriptor("_batch_size", "_state_dim")
    estimated_state_history = BatchTensorDescriptor(
        "_batch_size", "_max_history_length", "_state_dim"
    )
    # emission_difference_history = BatchTensorDescriptor(
    #     "_batch_size", "_history_length", "_state_dim"
    # )
    emission_history = BatchTensorDescriptor(
        "_batch_size", "_max_history_length", "_state_dim"
    )
    control_history = BatchTensorDescriptor(
        "_batch_size", "_max_history_length", "_state_dim"
    )

    @torch.inference_mode()
    def get_emission_history(
        self,
    ) -> torch.Tensor:
        """
        Get the emission history.

        Returns:
        --------
        torch.Tensor
            Emission history of shape (batch_size, history_length, state_dim)
        """
        return self._emission_history[:, -self._history_length :, :]

    @torch.inference_mode()
    def update_history(
        self,
        emission: torch.Tensor,
        # estimated_distribution: torch.Tensor,
        # emission_difference: torch.Tensor,
    ) -> None:
        # logger.info(
        #     f"Updating history with emission shape: {emission.shape}"
        # )

        self._estimated_state_history = torch.roll(
            self._estimated_state_history, -1, dims=1
        )
        self._emission_history = torch.roll(self._emission_history, -1, dims=1)
        self._emission_history[:, -1, :] = emission[:, 0, :]

        # logger.info(
        #     f"Updating history with emission shape: {self._emission_history.shape}"
        # )

        self._history_length = min(
            self._history_length + 1, self._max_history_length
        )

        # self._estimated_state_history[:, -1, :] = estimated_distribution[
        #     :, 0, :
        # ]
        # self._emission_difference_history = torch.roll(
        #     self._emission_difference_history, -1, dims=1
        # )
        # self._emission_difference_history[:, -1, :] = emission_difference[
        #     :, 0, :
        # ]

    # def compute_emission_difference(
    #     self,
    #     emission: torch.Tensor,
    # ) -> torch.Tensor:
    #     """
    #     Compute the emission difference.

    #     Arguments:
    #     ----------
    #     emission : torch.Tensor
    #         Emission tensor of shape (batch_size, discrete_observation_dim)

    #     Returns:
    #     --------
    #     torch.Tensor
    #         Emission difference of shape (batch_size, state_dim)
    #     """
    #     return emission * 2 - 1
