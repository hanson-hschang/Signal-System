from typing import Callable, Generic, Tuple, cast

from dataclasses import dataclass, field

from ss.estimation.filtering.hmm.learning.module.emission.config import (
    EmissionConfig,
)
from ss.estimation.filtering.hmm.learning.module.estimation.config import (
    EstimationConfig,
)
from ss.estimation.filtering.hmm.learning.module.filter.config import (
    FilterConfig,
)
from ss.estimation.filtering.hmm.learning.module.transition.block.config import (
    TransitionBlockConfig,
)
from ss.estimation.filtering.hmm.learning.module.transition.config import (
    TransitionConfig,
)
from ss.estimation.filtering.hmm.learning.module.transition.layer.config import (
    TransitionLayerConfig,
)
from ss.utility.assertion.validator import PositiveIntegerValidator
from ss.utility.descriptor import DataclassDescriptor
from ss.utility.learning.module.config import BaseLearningConfig
from ss.utility.learning.parameter.probability.config import (
    ProbabilityParameterConfig,
)
from ss.utility.learning.parameter.transformer.config import TC
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


@dataclass
class LearningHmmFilterConfig(BaseLearningConfig, Generic[TC]):
    """
    Configuration of the `LearningHmmFilter` class.
    """

    filter: FilterConfig = field(
        default_factory=cast(
            Callable[[], FilterConfig],
            FilterConfig,
        )
    )
    transition: TransitionConfig[TC] = field(
        default_factory=TransitionConfig[TC]
    )
    emission: EmissionConfig[TC] = field(default_factory=EmissionConfig[TC])
    estimation: EstimationConfig[TC] = field(
        default_factory=EstimationConfig[TC]
    )
    # prediction: PredictionConfig = field(default_factory=PredictionConfig)

    @classmethod
    def basic_config(
        cls,
        state_dim: int,
        discrete_observation_dim: int,
        estimation_dim: int = 0,
        dropout_rate: float = 0.0,
        probability_option: ProbabilityParameterConfig.Option = (
            ProbabilityParameterConfig.Option.SOFTMAX
        ),
    ) -> "LearningHmmFilterConfig[TC]":
        """
        Create a basic configuration of the `LearningHmmFilter` module.

        Arguments
        ----------
        state_dim : int
            The dimension of the state.
        discrete_observation_dim : int
            The dimension of the discrete observation.
        estimation_dim : int
            The dimension of the estimation.
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
        # _block_dims = (
        #     (block_dims,)
        #     if isinstance(block_dims, int)
        #     else tuple(
        #         PositiveIntegerValidator(block_dim).get_value()
        #         for block_dim in block_dims
        #     )
        # )

        # Prepare transition process configuration
        # layers = []
        # for block_dim in _block_dims:
        #     # layer = TransitionLayerConfig[TC]()
        #     blocks = []
        #     for _ in range(block_dim):
        #         # layer.blocks.append(TransitionBlockConfig[TC]())
        #         blocks.append(TransitionBlockConfig[TC]())
        #     layers.append(TransitionLayerConfig[TC](blocks=tuple(blocks)))

        # Prepare filter configuration
        filter_config = FilterConfig(
            state_dim=state_dim,
            discrete_observation_dim=discrete_observation_dim,
            estimation_dim=estimation_dim,
        )

        # Prepare module configuration
        config = cls(
            filter=filter_config,
        )

        # Update estimation configuration
        if estimation_dim > 0:
            config.estimation.option = (
                EstimationConfig.Option.LINEAR_TRANSFORM_ESTIMATION
            )

        # Update probability parameter configuration
        config.emission.matrix.probability_parameter = (
            ProbabilityParameterConfig[TC].from_option(probability_option)
        )
        config.estimation.matrix.probability_parameter = (
            ProbabilityParameterConfig[TC].from_option(probability_option)
        )
        config.transition.initial_state.probability_parameter = (
            ProbabilityParameterConfig[TC].from_option(probability_option)
        )
        config.transition.matrix.probability_parameter = (
            ProbabilityParameterConfig[TC].from_option(probability_option)
        )

        # for layer in config.transition.layers:
        #     layer.coefficient.probability_parameter = (
        #         ProbabilityParameterConfig[TC].from_option(probability_option)
        #     )
        #     layer.step.initial_state.probability_parameter = (
        #         ProbabilityParameterConfig[TC].from_option(probability_option)
        #     )
        #     for block in layer.blocks:
        #         block.step.initial_state.probability_parameter = (
        #             ProbabilityParameterConfig[TC].from_option(
        #                 probability_option
        #             )
        #         )
        #         block.matrix.probability_parameter = (
        #             ProbabilityParameterConfig[TC].from_option(
        #                 probability_option
        #             )
        #         )

        # Update dropout configuration
        config.emission.matrix.probability_parameter.dropout.rate = (
            dropout_rate
        )
        config.estimation.matrix.probability_parameter.dropout.rate = (
            dropout_rate
        )
        config.transition.initial_state.probability_parameter.dropout.rate = (
            dropout_rate
        )
        config.transition.matrix.probability_parameter.dropout.rate = (
            dropout_rate
        )
        # for layer in config.transition.layers:
        #     layer.coefficient.probability_parameter.dropout.rate = dropout_rate
        #     layer.step.initial_state.probability_parameter.dropout.rate = (
        #         dropout_rate
        #     )
        #     for block in layer.blocks:
        #         block.step.initial_state.probability_parameter.dropout.rate = (
        #             dropout_rate
        #         )
        #         block.matrix.probability_parameter.dropout.rate = dropout_rate

        return config
