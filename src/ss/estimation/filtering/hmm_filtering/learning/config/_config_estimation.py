from typing import Optional

from dataclasses import dataclass
from enum import StrEnum

from ss.learning.config import BaseLearningConfig


@dataclass
class EstimationConfig(BaseLearningConfig):

    class Option(StrEnum):
        ESTIMATED_STATE = "ESTIMATED_STATE"
        PREDICTED_NEXT_STATE = "PREDICTED_NEXT_STATE"
        PREDICTED_NEXT_OBSERVATION_PROBABILITY = (
            "PREDICTED_NEXT_OBSERVATION_PROBABILITY"
        )
        PREDICTED_NEXT_STATE_OVER_LAYERS = "PREDICTED_NEXT_STATE_OVER_LAYERS"
        PREDICTED_NEXT_OBSERVATION_PROBABILITY_OVER_LAYERS = (
            "PREDICTED_NEXT_OBSERVATION_PROBABILITY_OVER_LAYERS"
        )

    option: Option = Option.ESTIMATED_STATE
    output_dim: Optional[int] = None
