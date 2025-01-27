from typing import Optional, Tuple

from dataclasses import dataclass
from enum import StrEnum

# from ss.estimation.filtering.hmm_filtering._hmm_filtering_learning_transition_block import (
#     LearningHmmFilterTransitionBlockOption,
# )
from ss.learning import BaseLearningConfig
from ss.utility.assertion.validator import PositiveIntegerValidator
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class LearningHmmFilterEstimationOption(StrEnum):
    """
    Enumeration class for the estimation option of the `LearningHmmFilterConfig` class.
    """

    ESTIMATED_STATE = "ESTIMATED_STATE"
    PREDICTED_NEXT_STATE = "PREDICTED_NEXT_STATE"
    PREDICTED_NEXT_OBSERVATION_PROBABILITY = (
        "PREDICTED_NEXT_OBSERVATION_PROBABILITY"
    )
    PREDICTED_NEXT_STATE_OVER_LAYERS = "PREDICTED_NEXT_STATE_OVER_LAYERS"
    PREDICTED_NEXT_OBSERVATION_PROBABILITY_OVER_LAYERS = (
        "PREDICTED_NEXT_OBSERVATION_PROBABILITY_OVER_LAYERS"
    )


class LearningHmmFilterTransitionBlockOption(StrEnum):
    FULL_MATRIX = "FULL_MATRIX"
    SPATIAL_INVARIANT = "SPATIAL_INVARIANT"

    # @classmethod
    # def get_block(
    #     cls,
    #     feature_id: int,
    #     state_dim: int,
    #     block_option: "LearningHmmFilterTransitionBlockOption",
    # ) -> BaseLearningHmmFilterTransitionBlock:
    #     match block_option:
    #         case cls.FULL_MATRIX:
    #             return LearningHmmFilterTransitionFullMatrixBlock(feature_id, state_dim)
    #         case cls.SPATIAL_INVARIANT:
    #             return LearningHmmFilterTransitionSpatialInvariantBlock(
    #                 feature_id, state_dim
    #             )
    #         case _ as _invalid_block_option:
    #             assert_never(_invalid_block_option)


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
    feature_dim_over_layers : Optional[Tuple[int, ...]], default = None
        The dimension of the feature over layers.
        The length of the tuple is the number of layers.
        The values of the tuple (positive integers) are the dimension of features for each layer.
    feature_dim : Optional[int], default = None
        The dimension of the feature for all layers.
        If `feature_dim_over_layers` is not None, this value is ignored.
    layer_dim : Optional[int], default = None
        The dimension of the layer.
        If `feature_dim_over_layers` is not None, this value is ignored.
    dropout_rate : float, default = 0.1
        The dropout rate for the model. (0.0 <= dropout_rate < 1.0)
    block_option : LearningHiddenMarkovModelFilterBlockOption, default = LearningHiddenMarkovModelFilterBlockOptions.FULL_MATRIX
        The block option for the model.
    """

    state_dim: int
    discrete_observation_dim: int
    feature_dim_over_layers: Optional[Tuple[int, ...]] = None
    feature_dim: Optional[int] = None
    layer_dim: Optional[int] = None
    dropout_rate: float = 0.1
    block_option: LearningHmmFilterTransitionBlockOption = (
        LearningHmmFilterTransitionBlockOption.FULL_MATRIX
    )
    estimation_option: LearningHmmFilterEstimationOption = (
        LearningHmmFilterEstimationOption.PREDICTED_NEXT_OBSERVATION_PROBABILITY
    )

    def __post_init__(self) -> None:
        self.state_dim = PositiveIntegerValidator(self.state_dim).get_value()
        self.discrete_observation_dim = PositiveIntegerValidator(
            self.discrete_observation_dim
        ).get_value()
        assert 0.0 <= self.dropout_rate < 1.0, (
            f"dropout_rate must be in the range of [0.0, 1.0). "
            f"dropout_rate given is {self.dropout_rate}."
        )

        # Check the consistency of feature_dim_over_layers, feature_dim, and layer_dim
        if self.feature_dim_over_layers is None:
            if self.layer_dim is None:
                self.layer_dim = 1
            if self.feature_dim is None:
                self.feature_dim = 1
            self._feature_dim_over_layers = tuple(
                [self.feature_dim] * self.layer_dim
            )
        else:
            # Check the consistency of feature_dim
            feature_dim = self.feature_dim_over_layers[0]
            for i in range(len(self.feature_dim_over_layers)):
                assert type(self.feature_dim_over_layers[i]) == int, (
                    f"feature_dim_over_layers must be a tuple of integers. "
                    f"feature_dim_over_layers given is {self.feature_dim_over_layers}."
                )
                if (
                    feature_dim > 0
                    and self.feature_dim_over_layers[i] != feature_dim
                ):
                    feature_dim = -1
            self._feature_dim_over_layers = tuple(self.feature_dim_over_layers)

            if self.feature_dim is not None:
                if (feature_dim > 0) and (self.feature_dim != feature_dim):
                    logger.warning(
                        "Input argument `feature_dim` is ignored. "
                        "`feature_dim` is reset to the value in `feature_dim_over_layers`."
                    )
                    self.feature_dim = feature_dim
                if feature_dim < 0:
                    logger.warning(
                        "Input argument `feature_dim` is ignored. "
                        "All values in `feature_dim_over_layers` are not the same. "
                        "`feature_dim` is reset to None."
                    )
                    self.feature_dim = None

            # Check the consistency of layer_dim
            layer_dim = len(self.feature_dim_over_layers)
            if self.layer_dim is not None and self.layer_dim != layer_dim:
                logger.warning(
                    f"Input argument `layer_dim` is ignored. "
                    f"`layer_dim` is set to the length of feature_dim_over_layers, which is {layer_dim}."
                )
            self.layer_dim = int(layer_dim)
        self._layer_dim = self.layer_dim

    def get_feature_dim(self, layer_id: int) -> int:
        return self._feature_dim_over_layers[layer_id]

    def get_layer_dim(self) -> int:
        return self._layer_dim
