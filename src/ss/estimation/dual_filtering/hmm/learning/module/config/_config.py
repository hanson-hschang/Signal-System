from typing import Callable, Generic, Tuple, cast

from dataclasses import dataclass, field

from ss.estimation.dual_filtering.hmm.learning.module.emission.config import (
    DualEmissionConfig,
)
from ss.estimation.dual_filtering.hmm.learning.module.estimation.config import (
    DualEstimationConfig,
)
from ss.estimation.dual_filtering.hmm.learning.module.filter.config import (
    DualFilterConfig,
)
from ss.estimation.dual_filtering.hmm.learning.module.transition.block.config import (
    DualTransitionBlockConfig,
)
from ss.estimation.dual_filtering.hmm.learning.module.transition.config import (
    DualTransitionConfig,
)
from ss.estimation.dual_filtering.hmm.learning.module.transition.layer.config import (
    DualTransitionLayerConfig,
)
from ss.utility.assertion.validator import PositiveIntegerValidator
from ss.utility.learning.module.config import BaseLearningConfig
from ss.utility.learning.parameter.probability.config import (
    ProbabilityParameterConfig,
)
from ss.utility.learning.parameter.transformer.config import TC
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


@dataclass
class LearningDualHmmFilterConfig(BaseLearningConfig, Generic[TC]):
    """
    Configuration of the `LearningHmmDualFilter` class.
    """

    filter: DualFilterConfig = field(
        default_factory=cast(
            Callable[[], DualFilterConfig],
            DualFilterConfig,
        )
    )
    transition: DualTransitionConfig[TC] = field(
        default_factory=cast(
            Callable[[], DualTransitionConfig[TC]],
            DualTransitionConfig[TC],
        )
    )
    emission: DualEmissionConfig[TC] = field(
        default_factory=DualEmissionConfig[TC]
    )
    estimation: DualEstimationConfig[TC] = field(
        default_factory=DualEstimationConfig[TC]
    )

    @classmethod
    def basic_config(
        cls,
        state_dim: int,
        discrete_observation_dim: int,
        estimation_dim: int = 0,
        history_length: int = 1,
        # block_dims: int | Tuple[int, ...] = 1,
        dropout_rate: float = 0.0,
        probability_option: ProbabilityParameterConfig.Option = (
            ProbabilityParameterConfig.Option.SOFTMAX
        ),
        # transition_matrix_binding: bool = True,
    ) -> "LearningDualHmmFilterConfig[TC]":
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
        #         blocks.append(DualTransitionBlockConfig[TC]())
        #     layers.append(DualTransitionLayerConfig[TC](blocks=tuple(blocks)))

        # Prepare filter configuration
        filter_config = DualFilterConfig(
            state_dim=state_dim,
            discrete_observation_dim=discrete_observation_dim,
            estimation_dim=estimation_dim,
            history_length=history_length,
        )

        # Prepare module configuration
        config = cls(
            filter=filter_config,
            # transition=DualTransitionConfig[TC](
            #     layers=tuple(layers),
            #     transition_matrix_binding=transition_matrix_binding,
            # ),
        )

        # Update estimation configuration
        if estimation_dim > 0:
            config.estimation.option = (
                DualEstimationConfig.Option.LINEAR_TRANSFORM_ESTIMATION
            )

        # Update probability parameter configuration
        config.emission.matrix.probability_parameter = (
            ProbabilityParameterConfig[TC].from_option(probability_option)
        )
        config.estimation.matrix.probability_parameter = (
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
