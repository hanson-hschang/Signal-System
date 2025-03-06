from typing import Optional

from dataclasses import dataclass
from enum import StrEnum, auto

from ss.utility.learning.module import config as Config


@dataclass
class EstimationConfig(Config.BaseLearningConfig):

    class Option(StrEnum):
        ESTIMATED_STATE = auto()
        PREDICTED_NEXT_STATE = auto()
        PREDICTED_NEXT_OBSERVATION_PROBABILITY = auto()
        PREDICTED_NEXT_STATE_OVER_LAYERS = auto()
        PREDICTED_NEXT_OBSERVATION_PROBABILITY_OVER_LAYERS = auto()

    option: Option = Option.ESTIMATED_STATE
    output_dim: Optional[int] = None
