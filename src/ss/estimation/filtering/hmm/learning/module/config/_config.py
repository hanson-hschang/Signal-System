from typing import Self, Tuple

from dataclasses import dataclass, field

from ss.utility.assertion.validator import PositiveIntegerValidator
from ss.utility.learning.module import config as Config
from ss.utility.learning.module.dropout.config import DropoutConfig
from ss.utility.logging import Logging

from ._config_emission import EmissionProcessConfig
from ._config_estimation import EstimationConfig
from ._config_filter import FilterConfig
from ._config_prediction import PredictionConfig
from ._config_transition import (
    TransitionBlockConfig,
    TransitionLayerConfig,
    TransitionProcessConfig,
)

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
    block_dim_over_layers : Tuple[int, ...]
        The dimension of blocks for each layer.
        The length of the tuple is the number of layers.
        The values of the tuple (positive integers) are the dimension of blocks for each layer.
    """

    # state_dim: int
    # discrete_observation_dim: int
    # feature_dim_over_layers: Tuple[int, ...]
    filter: FilterConfig = field(default_factory=lambda: FilterConfig())
    # dropout: DropoutConfig = field(default_factory=DropoutConfig)
    transition: TransitionProcessConfig = field(
        default_factory=lambda: TransitionProcessConfig()
    )
    emission: EmissionProcessConfig = field(
        default_factory=EmissionProcessConfig
    )
    estimation: EstimationConfig = field(default_factory=EstimationConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)

    # def __post_init__(self) -> None:
    #     self.state_dim = PositiveIntegerValidator(self.state_dim).get_value()
    #     self.discrete_observation_dim = PositiveIntegerValidator(
    #         self.discrete_observation_dim
    #     ).get_value()
    #     for feature_dim in self.feature_dim_over_layers:
    #         assert type(feature_dim) == int, (
    #             f"feature_dim_over_layers must be a tuple of integers. "
    #             f"feature_dim_over_layers given is {self.feature_dim_over_layers}."
    #         )

    # @property
    # def layer_dim(self) -> int:
    #     return len(self.feature_dim_over_layers)

    # def get_feature_dim(self, layer_id: int) -> int:
    #     return self.feature_dim_over_layers[layer_id]

    @classmethod
    def basic_config(
        cls,
        state_dim: int,
        discrete_observation_dim: int,
        block_dims: int | Tuple[int, ...],
    ) -> Self:
        filter_config = FilterConfig()
        filter_config.state_dim = state_dim
        filter_config.discrete_observation_dim = discrete_observation_dim

        config = cls(filter=filter_config)
        # config.transition.layers = [TransitionLayerConfig()]
        # print(type(config.transition.layers[0]))

        _block_dims = (
            (block_dims,)
            if isinstance(block_dims, int)
            else tuple(
                PositiveIntegerValidator(block_dim).get_value()
                for block_dim in block_dims
            )
        )

        for block_dim in _block_dims:
            layer = TransitionLayerConfig()
            for _ in range(block_dim):
                layer.blocks.append(TransitionBlockConfig())
            config.transition.layers.append(layer)

        return config
