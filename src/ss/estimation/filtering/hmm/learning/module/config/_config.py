from typing import Callable, Generic, Optional, Self, Tuple, TypeVar, cast

from dataclasses import dataclass, field

from ss.utility.assertion.validator import PositiveIntegerValidator
from ss.utility.descriptor import DataclassDescriptor
from ss.utility.learning.module.config import BaseLearningConfig
from ss.utility.learning.parameter.probability.config import (
    ProbabilityParameterConfig,
)
from ss.utility.learning.parameter.transformer.config import TC
from ss.utility.learning.parameter.transformer.norm.min_zero.config import (
    MinZeroNormTransformerConfig,
)

# from ss.utility.learning.parameter.transformer.softmax.config import SoftmaxTC, SoftmaxTransformerConfig
from ss.utility.learning.parameter.transformer.softmax.linear.config import (
    LinearSoftmaxTransformerConfig,
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


class FilterDescriptor(DataclassDescriptor[FilterConfig]):
    def __set__(
        self,
        obj: object,
        value: FilterConfig,
    ) -> None:
        assert isinstance(value, FilterConfig)
        super().__set__(obj, value)


class TransitionDescriptor(
    DataclassDescriptor[TransitionProcessConfig[TC]],
    Generic[TC],
):
    # def __init__(self, value: Optional[TransitionProcessConfig[TC_SOFTMAX]] = None):
    #     if value is None:
    #         value = TransitionProcessConfig[TC_SOFTMAX]()
    #     super().__init__(value)

    def __set__(
        self,
        obj: object,
        value: TransitionProcessConfig[TC],
    ) -> None:
        assert isinstance(value, TransitionProcessConfig)
        super().__set__(obj, value)


class EmissionDescriptor(
    DataclassDescriptor[EmissionProcessConfig[TC]], Generic[TC]
):
    def __set__(
        self,
        obj: object,
        value: EmissionProcessConfig[TC],
    ) -> None:
        assert isinstance(value, EmissionProcessConfig)
        super().__set__(obj, value)


@dataclass
class LearningHmmFilterConfig(BaseLearningConfig, Generic[TC]):
    """
    Configuration of the `LearningHmmFilter` class.
    """

    filter: FilterDescriptor = FilterDescriptor(
        field(default_factory=lambda: FilterConfig())
    )
    transition: TransitionDescriptor[TC] = TransitionDescriptor[TC](
        field(
            default_factory=cast(
                Callable[[], TransitionProcessConfig[TC]],
                TransitionProcessConfig[TC],
            )
        )
    )
    emission: EmissionDescriptor[TC] = EmissionDescriptor[TC](
        field(default_factory=lambda: EmissionProcessConfig[TC]())
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
        probability_option: ProbabilityParameterConfig.Option = (
            ProbabilityParameterConfig.Option.SOFTMAX
        ),
    ) -> "LearningHmmFilterConfig":
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
        dropout_rate : float
            The dropout rate.
        probability_option : ProbabilityParameterConfig.Option
            The option of the probability parameter.

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

        # Prepare transition process configuration
        layers = []
        for block_dim in _block_dims:
            # layer = TransitionLayerConfig[TC]()
            blocks = []
            for _ in range(block_dim):
                # layer.blocks.append(TransitionBlockConfig[TC]())
                blocks.append(TransitionBlockConfig[TC]())
            layers.append(TransitionLayerConfig[TC](blocks=tuple(blocks)))

        # Prepare filter configuration
        filter_config = FilterConfig()
        filter_config.state_dim = state_dim
        filter_config.discrete_observation_dim = discrete_observation_dim

        # Prepare module configuration
        config = cls(
            filter=filter_config,
            transition=TransitionProcessConfig[TC](layers=tuple(layers)),
            emission=EmissionProcessConfig[TC](),
        )

        # Update probability parameter configuration
        config.emission.block.matrix.probability_parameter = (
            ProbabilityParameterConfig[TC].from_option(probability_option)
        )
        for layer in config.transition.layers:
            layer.coefficient.probability_parameter = (
                ProbabilityParameterConfig[TC].from_option(probability_option)
            )
            layer.initial_state.probability_parameter = (
                ProbabilityParameterConfig[TC].from_option(probability_option)
            )
            for block in layer.blocks:
                block.initial_state.probability_parameter = (
                    ProbabilityParameterConfig[TC].from_option(
                        probability_option
                    )
                )
                block.matrix.probability_parameter = (
                    ProbabilityParameterConfig[TC].from_option(
                        probability_option
                    )
                )

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

        return config
