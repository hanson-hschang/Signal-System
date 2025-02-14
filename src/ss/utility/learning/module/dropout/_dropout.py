from typing import assert_never

import torch

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
        NonnegativeNumberValidator(rate := config.rate)
        super().__init__(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (self._config.rate == 0) or (x.numel() == 1):
            # If the model is not in training mode, the dropout rate is 0,
            # or the input tensor is a scalar, return the input tensor as it is.
            return x

        # keep _x at least a 2D or more dimensional tensor
        _x = (x.unsqueeze(0) if x.dim() == 1 else x).to(device=x.device)

        # Generate a mask tensor with the same shape as the input tensor _x
        # with a dynamic dropout rate in the range [0, self._config.rate)
        rate = torch.empty(1, device=_x.device).uniform_(0, self._config.rate)
        mask = torch.empty(_x.shape, device=_x.device).bernoulli_(1 - rate)

        match self._config.value:

            case self._config.Value.ZERO:
                result = _x * mask
                result = (
                    result / (1 - rate)
                    if self._config.value.scaling
                    else result
                )

            case self._config.Value.LOG_ZERO:
                # Add a constant 1 to the last dimension of the input tensor _x
                _x_shape = _x.shape
                extended_x_shape = _x_shape[:-1] + (_x_shape[-1] + 1,)
                extended_x = torch.empty(
                    extended_x_shape, dtype=_x.dtype, device=_x.device
                )
                extended_x[..., :-1] = _x
                extended_x[..., -1] = 1.0

                # Calculate the norm of the tensor
                # the result of the first norm calculation is a 1D tensor
                # expend the norm tensor to the same shape as the input tensor _x
                norm = torch.norm(
                    extended_x, p=2, dim=list(range(1, len(_x_shape)))
                ).to(device=_x.device)
                for _ in range(1, len(_x_shape)):
                    norm = norm.unsqueeze(-1)
                norm = norm.expand(_x_shape)

                # Calculate the result of the dropout
                result = (
                    mask * _x
                    + (1 - mask) * norm * self._config.value.log_zero_scale
                )
            case _ as _invalid_value:
                assert_never(_invalid_value)

        if x.dim() == 1:
            result = result.squeeze(0)
        return result
