from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, cast

from ss.estimation.filtering.hmm.learning.module.emission.config import (
    EmissionConfig,
)
from ss.estimation.filtering.hmm.learning.module.estimation.config import (
    EstimationConfig,
)
from ss.estimation.filtering.hmm.learning.module.filter.config import (
    DualFilterConfig,
    FilterConfig,
)
from ss.estimation.filtering.hmm.learning.module.transition.config import (
    TransitionConfig,
)
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

        # Prepare module configuration
        config = cls(
            filter=FilterConfig(
                state_dim=state_dim,
                discrete_observation_dim=discrete_observation_dim,
                estimation_dim=estimation_dim,
            ),
        )

        # Update estimation configuration
        if config.filter.estimation_dim > 0:
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
        config.transition.matrix.probability_parameter = (
            ProbabilityParameterConfig[TC].from_option(probability_option)
        )
        config.transition.initial_state.probability_parameter = (
            ProbabilityParameterConfig[TC].from_option(probability_option)
        )

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

        return config


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
    transition: TransitionConfig[TC] = field(
        default_factory=TransitionConfig[TC]
    )
    emission: EmissionConfig[TC] = field(default_factory=EmissionConfig[TC])
    estimation: EstimationConfig[TC] = field(
        default_factory=EstimationConfig[TC]
    )

    @classmethod
    def basic_config(
        cls,
        state_dim: int,
        discrete_observation_dim: int,
        estimation_dim: int = 0,
        history_horizon: int = 1,
        dropout_rate: float = 0.0,
        probability_option: ProbabilityParameterConfig.Option = (
            ProbabilityParameterConfig.Option.SOFTMAX
        ),
    ) -> "LearningDualHmmFilterConfig[TC]":
        """
        Create a basic configuration of the `LearningDualHmmFilter` module.

        Arguments
        ----------
        state_dim : int
            The dimension of the state.
        discrete_observation_dim : int
            The dimension of the discrete observation.
        estimation_dim : int
            The dimension of the estimation.
        history_horizon : int
            The history horizon.
        dropout_rate : float
            The dropout rate.
        probability_option : ProbabilityParameterConfig.Option
            The option of the probability parameter.

        Returns
        -------
        config: LearningDualHmmFilterConfig
            The basic configuration of the `LearningDualHmmFilter` module.
        """

        # Prepare module configuration
        config = cls(
            filter=DualFilterConfig(
                state_dim=state_dim,
                discrete_observation_dim=discrete_observation_dim,
                estimation_dim=estimation_dim,
                history_horizon=history_horizon,
            ),
        )

        # Update estimation configuration
        if config.filter.estimation_dim > 0:
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
        config.transition.matrix.probability_parameter = (
            ProbabilityParameterConfig[TC].from_option(probability_option)
        )
        config.transition.initial_state.probability_parameter = (
            ProbabilityParameterConfig[TC].from_option(probability_option)
        )

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

        return config
