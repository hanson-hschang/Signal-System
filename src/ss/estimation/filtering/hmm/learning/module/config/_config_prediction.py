from typing import Optional, assert_never

from dataclasses import dataclass
from enum import StrEnum, auto

import torch

from ss.utility.learning.module import config as Config
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


@dataclass
class PredictionConfig(Config.BaseLearningConfig):

    class Option(StrEnum):
        AS_IS = auto()
        TOP_K = auto()
        TOP_P = auto()

    temperature: Optional[float] = None
    option: Option = Option.AS_IS
    max_number_of_choices: Optional[int] = None
    probability_threshold: Optional[float] = None

    def process_probability(self, probability: torch.Tensor) -> torch.Tensor:
        """
        Process the probability tensor based on the configuration.

        Arguments
        ---------
        probability : torch.Tensor
            shape = (batch_size, number_of_choices)
            The probability tensor to process.

        Returns
        -------
        torch.Tensor
            The processed probability tensor.
        """
        if self.temperature is not None:
            if self.temperature > 0.0:
                probability = torch.nn.functional.softmax(
                    torch.log(probability) / self.temperature,
                    dim=-1,
                )
            else:
                raise ValueError("temperature must be positive.")
        match self.option:
            case self.Option.AS_IS:
                processed_probability = probability
            case self.Option.TOP_K:
                if self.max_number_of_choices is None:
                    raise ValueError("max_number_of_choices must be set.")
                if self.max_number_of_choices <= 0:
                    raise ValueError(
                        "max_number_of_choices must be a positive integer."
                    )
                number_of_choices = probability.size(dim=-1)
                if self.max_number_of_choices >= number_of_choices:
                    processed_probability = probability
                else:
                    _processed_probability = torch.topk(
                        probability, self.max_number_of_choices, dim=-1
                    )
                    processed_probability = torch.zeros_like(probability)
                    processed_probability = processed_probability.scatter_(
                        dim=-1,
                        index=_processed_probability.indices,
                        src=_processed_probability.values,
                    )
            case self.Option.TOP_P:
                if self.probability_threshold is None:
                    raise ValueError("probability_threshold must be set.")
                if (
                    self.probability_threshold <= 0.0
                    or self.probability_threshold >= 1.0
                ):
                    raise ValueError(
                        "probability_threshold must be in the range of (0.0, 1.0)."
                    )
                sorted_probability, indices = torch.sort(
                    probability, descending=True, dim=-1
                )
                cumulative_probability = torch.cumsum(
                    sorted_probability, dim=-1
                )
                mask = cumulative_probability <= self.probability_threshold
                zeros = torch.zeros_like(probability)
                processed_probability = torch.where(mask, probability, zeros)
                processed_probability = zeros.scatter_(
                    dim=-1, index=indices, src=processed_probability
                )
            case _ as _invalid_option:
                assert_never(_invalid_option)
        processed_probability = (
            processed_probability
            / processed_probability.sum(dim=-1, keepdim=True)
        )
        return processed_probability
