from typing import Tuple

from dataclasses import dataclass, field

from ss.utility.assertion.validator import PositiveIntegerValidator
from ss.utility.learning.module import config as Config


@dataclass
class FilterConfig(Config.BaseLearningConfig):
    """

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

    state_dim: int = 1
    discrete_observation_dim: int = 1
    feature_dim_over_layers: Tuple[int, ...] = field(
        default_factory=lambda: (1,)
    )

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
