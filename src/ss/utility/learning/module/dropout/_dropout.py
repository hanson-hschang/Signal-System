from typing import Callable, assert_never

import torch
import torch.nn as nn

from ss.utility.assertion.validator import NonnegativeNumberValidator
from ss.utility.learning.module import BaseLearningModule
from ss.utility.learning.module.dropout import config as Config


class Dropout(BaseLearningModule[Config.DropoutConfig]):
    """
    Dropout without rescaling and variable dropout rates.
    """

    def __init__(
        self,
        config: Config.DropoutConfig,
    ) -> None:
        super().__init__(config)
        self._max_rate = self._config.rate
        self._rate: torch.Tensor
        self._forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

        match self._config.value:
            case Config.DropoutConfig.Value.ZERO:
                self._scaling = self._config.scaling
                self._forward = self._forward_zero
            case Config.DropoutConfig.Value.LOG_ZERO:
                self._log_zero_scale = self._config.log_zero_scale
                self._forward = self._forward_log_zero
            case _ as _invalid_value:
                assert_never(_invalid_value)

    def _forward_zero(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        masked_x = x * mask
        result = masked_x / (1 - self._rate) if self._scaling else masked_x
        return result

    def _forward_log_zero(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # Add a constant 1 to the last dimension of the input tensor x
        x_shape = x.shape
        extended_x_shape = x_shape[:-1] + (x_shape[-1] + 1,)
        extended_x = torch.empty(
            extended_x_shape, dtype=x.dtype, device=x.device
        )
        extended_x[..., :-1] = x
        extended_x[..., -1] = 1.0

        # Calculate the norm of the tensor
        # the result of the first norm calculation is a 1D tensor
        # expend the norm tensor to the same shape as the input tensor x
        norm = torch.norm(
            extended_x, p=2, dim=list(range(1, len(x_shape)))
        ).to(device=x.device)
        for _ in range(1, len(x_shape)):
            norm = norm.unsqueeze(-1)
        norm = norm.expand(x_shape)

        # Calculate the result of the dropout
        result: torch.Tensor = (
            mask * x + (1 - mask) * norm * self._log_zero_scale
        )
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (self._max_rate == 0) or (x.numel() == 1):
            # If the model is not in training mode, the dropout rate is 0,
            # or the input tensor is a scalar, return the input tensor as it is.
            return x

        if isinstance(x, nn.Parameter) and x.requires_grad is False:
            # If the input tensor is a non-trainable parameter, return the input tensor as it is.
            return x

        # keep _x at least a 2D or more dimensional tensor
        _x = (x.unsqueeze(0) if x.dim() == 1 else x).to(device=x.device)

        # Generate a mask tensor with the same shape as the input tensor _x
        # with a dynamic dropout rate in the range [0, self._max_rate)
        self._rate = torch.empty(1, device=_x.device).uniform_(
            0, self._max_rate
        )
        mask = torch.empty(_x.shape, device=_x.device).bernoulli_(
            1 - self._rate
        )

        result = self._forward(_x, mask)

        if x.dim() == 1:
            result = result.squeeze(0)
        return result
