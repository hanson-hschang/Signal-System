from typing import Tuple

from dataclasses import dataclass, field

from ss.estimation.filtering.hmm_filtering.learning.config._config_emission import (
    EmissionConfig,
)
from ss.estimation.filtering.hmm_filtering.learning.config._config_estimation import (
    EstimationConfig,
)
from ss.estimation.filtering.hmm_filtering.learning.config._config_prediction import (
    PredictionConfig,
)
from ss.estimation.filtering.hmm_filtering.learning.config._config_transition import (
    TransitionConfig,
)
from ss.learning.config import BaseLearningConfig
from ss.utility.assertion.validator import PositiveIntegerValidator
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


@dataclass
class LearningHmmFilterConfig(BaseLearningConfig):
    """
    Configuration of the `LearningHmmFilter` class.

    Properties
    ----------
    state_dim : int
        The dimension of the state.
    discrete_observation_dim : int
        The dimension of the discrete observation.
    feature_dim_over_layers : Tuple[int, ...]
        The dimension of features for each layer.
        The length of the tuple is the number of layers.
        The values of the tuple (positive integers) are the dimension of features for each layer.
    dropout_rate : float, default = 0.1
        The dropout rate for the model. (0.0 <= dropout_rate < 1.0)
    """

    state_dim: int
    discrete_observation_dim: int
    feature_dim_over_layers: Tuple[int, ...]
    dropout_rate: float = 0.1
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    emission: EmissionConfig = field(default_factory=EmissionConfig)
    estimation: EstimationConfig = field(default_factory=EstimationConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)

    def __post_init__(self) -> None:
        self.state_dim = PositiveIntegerValidator(self.state_dim).get_value()
        self.discrete_observation_dim = PositiveIntegerValidator(
            self.discrete_observation_dim
        ).get_value()
        assert 0.0 <= self.dropout_rate < 1.0, (
            f"dropout_rate must be in the range of [0.0, 1.0). "
            f"dropout_rate given is {self.dropout_rate}."
        )
        for feature_dim in self.feature_dim_over_layers:
            assert type(feature_dim) == int, (
                f"feature_dim_over_layers must be a tuple of integers. "
                f"feature_dim_over_layers given is {self.feature_dim_over_layers}."
            )

    @property
    def layer_dim(self) -> int:
        return len(self.feature_dim_over_layers)

    def get_feature_dim(self, layer_id: int) -> int:
        return self.feature_dim_over_layers[layer_id]
