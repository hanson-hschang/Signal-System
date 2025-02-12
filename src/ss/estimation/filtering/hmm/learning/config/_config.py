from typing import Tuple

from dataclasses import dataclass, field

from ss.utility.assertion.validator import PositiveIntegerValidator
from ss.utility.learning import config as Config
from ss.utility.learning.module.dropout.config import DropoutConfig
from ss.utility.logging import Logging

from ._config_emission import EmissionConfig
from ._config_estimation import EstimationConfig
from ._config_prediction import PredictionConfig
from ._config_transition import TransitionConfig

logger = Logging.get_logger(__name__)


@dataclass
class LearningHmmFilterConfig(Config.BaseLearningConfig):
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
    """

    state_dim: int
    discrete_observation_dim: int
    feature_dim_over_layers: Tuple[int, ...]
    dropout: DropoutConfig = field(default_factory=DropoutConfig)
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    emission: EmissionConfig = field(default_factory=EmissionConfig)
    estimation: EstimationConfig = field(default_factory=EstimationConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)

    def __post_init__(self) -> None:
        self.state_dim = PositiveIntegerValidator(self.state_dim).get_value()
        self.discrete_observation_dim = PositiveIntegerValidator(
            self.discrete_observation_dim
        ).get_value()
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
