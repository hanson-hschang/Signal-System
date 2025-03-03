import numpy as np
import pytest
import torch

from ss.estimation.filtering.hmm.learning.module import LearningHmmFilter
from ss.estimation.filtering.hmm.learning.module import config as Config
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
    def config(self) -> Config.LearningHmmFilterConfig:
        LOG_ZERO_OFFSET = 16
        config = Config.LearningHmmFilterConfig.basic_config(
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
        transition_block_1.option = (
            Config.TransitionBlockConfig.Option.FULL_MATRIX
        )
        transition_block_2.option = (
            Config.TransitionBlockConfig.Option.SPATIAL_INVARIANT_MATRIX
        )
        for block in [transition_block_1, transition_block_2]:
            matrix_parameter = block.matrix.probability_parameter
            matrix_parameter.transformer.log_zero_offset = LOG_ZERO_OFFSET
            initial_state_parameter = block.initial_state.probability_parameter
            initial_state_parameter.transformer.log_zero_offset = (
                LOG_ZERO_OFFSET
            )

        emission_block = config.emission.block
        emission_matrix_parameter = emission_block.matrix.probability_parameter
        emission_matrix_parameter.transformer.log_zero_offset = LOG_ZERO_OFFSET

        return config

    @pytest.fixture
    def learning_filter(
        self, config: Config.LearningHmmFilterConfig
    ) -> LearningHmmFilter:
        learning_filter = LearningHmmFilter(config=config)
        transition_layer = learning_filter.transition_process.layers[0]
        with learning_filter.evaluation_mode():
            transition_layer.coefficient = torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
            )
            transition_layer.blocks[0].initial_state = torch.tensor(
                [1.0, 0.0, 0.0]
            )
            transition_layer.blocks[1].initial_state = torch.tensor(
                [0.0, 1.0, 0.0]
            )
            identity = torch.eye(learning_filter.state_dim)
            transition_layer.blocks[0].matrix = identity
            shifted_identity = torch.roll(identity, shifts=1, dims=1)
            transition_layer.blocks[1].matrix = shifted_identity
            learning_filter.emission_process.matrix = torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
            )
        return learning_filter

    @pytest.fixture
    def observation_trajectory(self) -> torch.Tensor:
        return torch.tensor([0, 0, 1])

    def test_learning_filter_parameter(
        self, learning_filter: LearningHmmFilter
    ) -> None:
        assert learning_filter.state_dim == 3
        assert learning_filter.discrete_observation_dim == 2
        assert learning_filter.emission_process.matrix.shape == (3, 2)
        transition_layer = learning_filter.transition_process.layers[0]
        transition_block_1 = transition_layer.blocks[0]
        transition_block_2 = transition_layer.blocks[1]
        for block in [transition_block_1, transition_block_2]:
            assert block.matrix.shape == (3, 3)
            assert block.initial_state.shape == (3,)
        emission_matrix = learning_filter.emission_process.matrix
        with learning_filter.evaluation_mode():
            assert_allclose(
                transition_layer.coefficient,
                np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
            )
            assert_allclose(
                transition_block_1.initial_state,
                np.array([1.0, 0.0, 0.0]),
            )
            assert_allclose(
                transition_block_2.initial_state,
                np.array([0.0, 1.0, 0.0]),
            )
            assert_allclose(
                transition_layer.matrix,
                np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.0, 0.5]]),
            )
            assert_allclose(
                transition_block_1.matrix,
                np.identity(3),
            )
            assert_allclose(
                transition_block_2.matrix,
                np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
            )
            assert_allclose(
                emission_matrix,
                np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
            )

    def test_learning_filter_estimation(
        self,
        learning_filter: LearningHmmFilter,
        observation_trajectory: torch.Tensor,
    ) -> None:
        ERROR_TOLERANCE = 1e-6
        learning_filter.estimation_option = (
            Config.EstimationConfig.Option.PREDICTED_NEXT_OBSERVATION_PROBABILITY_OVER_LAYERS
        )
        with BaseLearningProcess.inference_mode(learning_filter):
            transition_block_1 = learning_filter.transition_process.layers[
                0
            ].blocks[0]
            transition_block_2 = learning_filter.transition_process.layers[
                0
            ].blocks[1]

            learning_filter.reset()
            assert_allclose(
                transition_block_1.estimated_previous_state,
                np.array([1.0, 0.0, 0.0]),
            )
            assert_allclose(
                transition_block_2.estimated_previous_state,
                np.array([0.0, 1.0, 0.0]),
            )
            assert_allclose(
                transition_block_1.predicted_state,
                np.array([1.0, 0.0, 0.0]),
            )
            assert_allclose(
                transition_block_2.predicted_state,
                np.array([0.0, 0.0, 1.0]),
            )

            learning_filter.update(observation_trajectory[0])
            assert_allclose(
                transition_block_1.estimated_previous_state,
                np.array([1.0, 0.0, 0.0]),
            )
            assert_allclose(
                transition_block_2.estimated_previous_state,
                np.array([0.0, 0.0, 1.0]),
            )
            assert_allclose(
                transition_block_1.predicted_state,
                np.array([1.0, 0.0, 0.0]),
            )
            assert_allclose(
                transition_block_2.predicted_state,
                np.array([1.0, 0.0, 0.0]),
            )
            assert_allclose(
                learning_filter.estimate(),
                np.array([[5.0 / 6.0, 1.0 / 6.0], [1.0, 0.0]]),
            )

            learning_filter.update(observation_trajectory[1])
            assert_allclose(
                transition_block_1.estimated_previous_state,
                np.array([1.0, 0.0, 0.0]),
            )
            assert_allclose(
                transition_block_2.estimated_previous_state,
                np.array([1.0, 0.0, 0.0]),
            )
            assert_allclose(
                transition_block_1.predicted_state,
                np.array([1.0, 0.0, 0.0]),
            )
            assert_allclose(
                transition_block_2.predicted_state,
                np.array([0.0, 1.0, 0.0]),
            )
            # assert_allclose(
            #     learning_filter.estimate(),
            #     np.array([[1.0 / 6.0, 5.0 / 6.0], [1.0, 0.0]]),
            # )
