import torch

from ss.utility.learning.parameter.transformer import Transformer
from ss.utility.learning.parameter.transformer.exp import config as Config


class ExpTransformer(
    Transformer[Config.ExpTransformerConfig],
):
    def forward(self, parameter: torch.Tensor) -> torch.Tensor:
        return torch.exp(parameter)

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        return torch.log(value)
