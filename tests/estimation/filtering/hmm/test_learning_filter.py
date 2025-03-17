from typing import cast

import numpy as np
import pytest
import torch

from ss.estimation.filtering.hmm.learning.module import LearningHmmFilter
from ss.estimation.filtering.hmm.learning.module.config import (
    LearningHmmFilterConfig,
)
from ss.utility.learning.process import BaseLearningProcess


def assert_allclose(tensor: torch.Tensor, ndarray: np.ndarray) -> None:
    ERROR_TOLERANCE = 1e-6
    np.testing.assert_allclose(
        tensor.detach().numpy(),
        ndarray,
        atol=ERROR_TOLERANCE,
        rtol=ERROR_TOLERANCE,
    )


class TestLearningHmmFilter:

    @pytest.fixture
    def config(self) -> LearningHmmFilterConfig:
        from ss.utility.learning.parameter.transformer.softmax.config import (
            SoftmaxTransformerConfig,
        )

        LOG_ZERO_OFFSET = 16
        config = LearningHmmFilterConfig[
            SoftmaxTransformerConfig
        ].basic_config(
            state_dim=3,
            discrete_observation_dim=2,
            block_dims=(2,),
        )
        transition_layer = config.transition.layers[0]
        transition_coefficient_parameter = (
            transition_layer.coefficient.probability_parameter
        )
        transition_coefficient_parameter.transformer.log_zero_offset = (
            LOG_ZERO_OFFSET
        )

        transition_block_1 = config.transition.layers[0].blocks[0]
        transition_block_2 = config.transition.layers[0].blocks[1]
        transition_block_1.option = transition_block_1.Option.FULL_MATRIX
        transition_block_2.option = (
            transition_block_2.Option.SPATIAL_INVARIANT_MATRIX
        )
        for block in [transition_block_1, transition_block_2]:
            matrix_parameter = block.matrix.probability_parameter
            matrix_parameter.transformer.log_zero_offset = LOG_ZERO_OFFSET
            initial_state_parameter = (
                block.step.initial_state.probability_parameter
            )
            initial_state_parameter.transformer.log_zero_offset = (
                LOG_ZERO_OFFSET
            )

        emission_block = config.emission.block
        emission_matrix_parameter = emission_block.matrix.probability_parameter
        emission_matrix_parameter.transformer.log_zero_offset = LOG_ZERO_OFFSET

        return config

    @pytest.fixture
    def learning_filter_with_block_initial_state_binding(
        self, config: LearningHmmFilterConfig
    ) -> LearningHmmFilter:
        from ss.utility.learning.parameter.transformer.softmax import (
            SoftmaxTransformer,
        )
        from ss.utility.learning.parameter.transformer.softmax.config import (
            SoftmaxTransformerConfig,
        )

        learning_filter = LearningHmmFilter[
            SoftmaxTransformer, SoftmaxTransformerConfig
        ](config=config)
        transition_layer = learning_filter.transition.layers[0]
        with learning_filter.evaluation_mode():
            transition_layer.coefficient = torch.tensor([0.75, 0.25])
            transition_layer.blocks[0].initial_state = torch.tensor(
                [1.0, 0.0, 0.0]
            )
            transition_layer.blocks[1].initial_state = torch.tensor(
                [0.0, 1.0, 0.0]
            )
            identity = np.identity(learning_filter.state_dim)
            transition_layer.blocks[0].matrix = torch.tensor(
                np.fliplr(identity).copy()
            )
            shifted_identity = torch.roll(
                torch.tensor(identity), shifts=1, dims=1
            )
            transition_layer.blocks[1].matrix = shifted_identity
            learning_filter.emission.matrix = torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
            )
        return learning_filter

    @pytest.fixture
    def learning_filter_without_block_initial_state_binding(
        self, config: LearningHmmFilterConfig
    ) -> LearningHmmFilter:
        from ss.utility.learning.parameter.transformer.softmax import (
            SoftmaxTransformer,
        )
        from ss.utility.learning.parameter.transformer.softmax.config import (
            SoftmaxTransformerConfig,
        )

        config.transition.layers[0].block_state_binding = False
        learning_filter = LearningHmmFilter[
            SoftmaxTransformer, SoftmaxTransformerConfig
        ](config=config)
        transition_layer = learning_filter.transition.layers[0]
        with learning_filter.evaluation_mode():
            transition_layer.coefficient = torch.tensor([0.75, 0.25])
            transition_layer.blocks[0].initial_state = torch.tensor(
                [1.0, 0.0, 0.0]
            )
            transition_layer.blocks[1].initial_state = torch.tensor(
                [0.0, 1.0, 0.0]
            )
            identity = np.identity(learning_filter.state_dim)
            transition_layer.blocks[0].matrix = torch.tensor(
                np.fliplr(identity).copy()
            )
            shifted_identity = torch.roll(
                torch.tensor(identity), shifts=1, dims=1
            )
            transition_layer.blocks[1].matrix = shifted_identity
            learning_filter.emission.matrix = torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
            )
        return learning_filter

    @pytest.fixture
    def observation_trajectory(self) -> torch.Tensor:
        return torch.tensor([0, 0, 1])

    def test_learning_filter_parameter(
        self,
        learning_filter_without_block_initial_state_binding: LearningHmmFilter,
    ) -> None:
        learning_filter = learning_filter_without_block_initial_state_binding
        assert learning_filter.state_dim == 3
        assert learning_filter.discrete_observation_dim == 2
        assert learning_filter.emission.matrix.shape == (3, 2)
        transition_layer = learning_filter.transition.layers[0]
        transition_block_1 = transition_layer.blocks[0]
        transition_block_2 = transition_layer.blocks[1]
        for block in [transition_block_1, transition_block_2]:
            assert block.matrix.shape == (3, 3)
            assert block.initial_state.shape == (3,)
        with torch.compiler.set_stance("force_eager"):
            with BaseLearningProcess.inference_mode(learning_filter):
                assert_allclose(
                    transition_layer.coefficient,
                    np.array([0.75, 0.25]),
                )
                assert_allclose(
                    transition_block_1.initial_state,
                    np.array([1.0, 0.0, 0.0]),
                )
                assert_allclose(
                    transition_block_2.initial_state,
                    np.array([0.0, 1.0, 0.0]),
                )
                # assert_allclose(
                #     transition_layer.matrix,
                #     np.array(
                #         [[0.0, 0.25, 0.75], [0.0, 0.75, 0.25], [1.0, 0.0, 0.0]]
                #     ),
                # )
                assert_allclose(
                    transition_block_1.matrix,
                    np.fliplr(np.identity(3)),
                )
                assert_allclose(
                    transition_block_2.matrix,
                    np.array(
                        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
                    ),
                )
                assert_allclose(
                    learning_filter.emission.matrix,
                    np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
                )

    def test_learning_filter_estimation(
        self,
        learning_filter_without_block_initial_state_binding: LearningHmmFilter,
        observation_trajectory: torch.Tensor,
    ) -> None:
        learning_filter = learning_filter_without_block_initial_state_binding
        # learning_filter.estimation_option = (
        #     Config.EstimationConfig.Option.PREDICTED_NEXT_OBSERVATION_PROBABILITY_OVER_LAYERS
        # )
        with torch.compiler.set_stance("force_eager"):
            with BaseLearningProcess.inference_mode(learning_filter):
                transition_block_1 = learning_filter.transition.layers[
                    0
                ].blocks[0]
                transition_block_2 = learning_filter.transition.layers[
                    0
                ].blocks[1]

                learning_filter.reset()
                assert_allclose(
                    transition_block_1.estimated_state,
                    np.array([1.0, 0.0, 0.0]),
                )
                assert_allclose(
                    transition_block_2.estimated_state,
                    np.array([0.0, 1.0, 0.0]),
                )
                assert_allclose(
                    transition_block_1.predicted_next_state,
                    np.array([0.0, 0.0, 1.0]),
                )
                assert_allclose(
                    transition_block_2.predicted_next_state,
                    np.array([0.0, 0.0, 1.0]),
                )

                learning_filter.update(observation_trajectory[0])
                assert_allclose(
                    transition_block_1.estimated_state,
                    np.array([0.0, 0.0, 1.0]),
                )
                assert_allclose(
                    transition_block_2.estimated_state,
                    np.array([0.0, 0.0, 1.0]),
                )
                assert_allclose(
                    transition_block_1.predicted_next_state,
                    np.array([1.0, 0.0, 0.0]),
                )
                assert_allclose(
                    transition_block_2.predicted_next_state,
                    np.array([1.0, 0.0, 0.0]),
                )
                # assert_allclose(
                #     learning_filter.estimate(),
                #     np.array([[5.0 / 6.0, 1.0 / 6.0], [1.0, 0.0]]),
                # )
                assert_allclose(
                    learning_filter.estimate(),
                    np.array([1.0, 0.0]),
                )

                learning_filter.update(observation_trajectory[1])
                assert_allclose(
                    transition_block_1.estimated_state,
                    np.array([1.0, 0.0, 0.0]),
                )
                assert_allclose(
                    transition_block_2.estimated_state,
                    np.array([1.0, 0.0, 0.0]),
                )
                assert_allclose(
                    transition_block_1.predicted_next_state,
                    np.array([0.0, 0.0, 1.0]),
                )
                assert_allclose(
                    transition_block_2.predicted_next_state,
                    np.array([0.0, 1.0, 0.0]),
                )
                # assert_allclose(
                #     learning_filter.estimate(),
                #     np.array([[5.0 / 6.0, 1.0 / 6.0], [3.0 / 8.0, 5.0 / 8.0]]),
                # )
                assert_allclose(
                    learning_filter.estimate(),
                    np.array([3.0 / 8.0, 5.0 / 8.0]),
                )

                learning_filter.update(observation_trajectory[2])
                assert_allclose(
                    transition_block_1.estimated_state,
                    np.array([0.0, 0.0, 1.0]),
                )
                assert_allclose(
                    transition_block_2.estimated_state,
                    np.array([0.0, 1.0, 0.0]),
                )
                assert_allclose(
                    transition_block_1.predicted_next_state,
                    np.array([1.0, 0.0, 0.0]),
                )
                assert_allclose(
                    transition_block_2.predicted_next_state,
                    np.array([0.0, 0.0, 1.0]),
                )
                # assert_allclose(
                #     learning_filter.estimate(),
                #     np.array([[1.0 / 6.0, 5.0 / 6.0], [7.0 / 8.0, 1.0 / 8.0]]),
                # )
                assert_allclose(
                    learning_filter.estimate(),
                    np.array([7.0 / 8.0, 1.0 / 8.0]),
                )
