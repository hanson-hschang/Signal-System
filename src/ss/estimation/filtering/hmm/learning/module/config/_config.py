from typing import Callable, Generic, Optional, Self, Tuple, TypeVar

from dataclasses import dataclass, field

from ss.utility.assertion.validator import PositiveIntegerValidator
from ss.utility.learning.module import config as Config
from ss.utility.learning.parameter.probability.config import (
    ProbabilityParameterConfig,
)
from ss.utility.learning.parameter.transformer.config import TransformerConfig
from ss.utility.learning.parameter.transformer.softmax.config import (
    SoftmaxTransformerConfig,
)
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

TC = TypeVar("TC", bound=TransformerConfig, default=SoftmaxTransformerConfig)


@dataclass
class LearningHmmFilterConfig(Config.BaseLearningConfig, Generic[TC]):
    """
    Configuration of the `LearningHmmFilter` class.
    """

    filter: FilterConfig = field(default_factory=lambda: FilterConfig())
    # dropout: DropoutConfig = field(default_factory=DropoutConfig)
    emission: EmissionProcessConfig = field(
        default_factory=lambda: EmissionProcessConfig()
    )
    transition: TransitionProcessConfig = field(
        default_factory=lambda: TransitionProcessConfig()
    )
    estimation: EstimationConfig = field(
        default_factory=lambda: EstimationConfig()
    )
    prediction: PredictionConfig = field(
        default_factory=lambda: PredictionConfig()
    )

    @classmethod
    def basic_config(
        cls,
        state_dim: int,
        discrete_observation_dim: int,
        block_dims: int | Tuple[int, ...] = 1,
        dropout_rate: float = 0.0,
        probability_parameter_factory: Optional[
            Callable[[], ProbabilityParameterConfig[TC]]
        ] = None,
    ) -> Self:
        """
        Create a basic configuration of the `LearningHmmFilter` module.

        Arguments
        ----------
        state_dim : int
            The dimension of the state.
        discrete_observation_dim : int
            The dimension of the discrete observation.
        block_dims : int | Tuple[int, ...]
            The dimension of blocks for each layer.
            The length of the tuple is the number of layers.
            The values of the tuple (positive integers) are the dimension of blocks for each layer.

        Returns
        -------
        config: LearningHmmFilterConfig
            The basic configuration of the `LearningHmmFilter` module.
        """
        # Validate block_dims
        _block_dims = (
            (block_dims,)
            if isinstance(block_dims, int)
            else tuple(
                PositiveIntegerValidator(block_dim).get_value()
                for block_dim in block_dims
            )
        )

        # Prepare filter configuration
        filter_config = FilterConfig()
        filter_config.state_dim = state_dim
        filter_config.discrete_observation_dim = discrete_observation_dim

        # Prepare module configuration
        config = cls(filter=filter_config)

        # Update transition process' configuration
        for block_dim in _block_dims:
            layer = TransitionLayerConfig()
            for _ in range(block_dim):
                layer.blocks.append(TransitionBlockConfig())
            config.transition.layers.append(layer)

        # Update dropout configuration
        config.emission.block.matrix.probability_parameter.dropout.rate = (
            dropout_rate
        )
        for layer in config.transition.layers:
            layer.coefficient.probability_parameter.dropout.rate = dropout_rate
            layer.initial_state.probability_parameter.dropout.rate = (
                dropout_rate
            )
            for block in layer.blocks:
                block.initial_state.probability_parameter.dropout.rate = (
                    dropout_rate
                )
                block.matrix.probability_parameter.dropout.rate = dropout_rate

        # Update probability parameter configuration
        if probability_parameter_factory is not None:
            config.emission.block.matrix.probability_parameter = (
                probability_parameter_factory()
            )
            for layer in config.transition.layers:
                layer.coefficient.probability_parameter = (
                    probability_parameter_factory()
                )
                layer.initial_state.probability_parameter = (
                    probability_parameter_factory()
                )
                for block in layer.blocks:
                    block.initial_state.probability_parameter = (
                        probability_parameter_factory()
                    )
                    block.matrix.probability_parameter = (
                        probability_parameter_factory()
                    )

        return config
