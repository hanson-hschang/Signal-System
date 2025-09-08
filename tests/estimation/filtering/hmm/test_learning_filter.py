from typing import cast

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from ss.estimation.filtering.hmm.learning.module import LearningHmmFilter
from ss.estimation.filtering.hmm.learning.module.config import (
    LearningHmmFilterConfig,
)
from ss.utility.learning.compile import CompileContext
from ss.utility.learning.compile.config import CompileConfig
from ss.utility.learning.process import BaseLearningProcess


def assert_allclose(tensor: torch.Tensor, ndarray: NDArray) -> None:
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
        )
        matrix_parameter = config.transition.matrix.probability_parameter
        initial_state_parameter = (
            config.transition.initial_state.probability_parameter
        )
        emission_matrix_parameter = (
            config.emission.matrix.probability_parameter
        )

        for parameter in (
            matrix_parameter,
            initial_state_parameter,
            emission_matrix_parameter,
        ):
            parameter.transformer.log_zero_offset = LOG_ZERO_OFFSET

        # initial_state_parameter.transformer.log_zero_offset = (
        #     LOG_ZERO_OFFSET
        # )
        # matrix_parameter.transformer.log_zero_offset = LOG_ZERO_OFFSET
        # emission_matrix_parameter.transformer.log_zero_offset = LOG_ZERO_OFFSET

        return config

    @pytest.fixture
    def learning_filter(
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
        with learning_filter.evaluation_mode():
            learning_filter.transition.initial_state = torch.tensor(
                [1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0]
            )
            learning_filter.transition.matrix = torch.tensor(
                [[0.75, 0.25, 0.0], [0.0, 0.75, 0.25], [0.25, 0.0, 0.75]]
            )
            learning_filter.emission.matrix = torch.tensor(
                [[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]]
            )
        return learning_filter

    @pytest.fixture
    def observation_trajectory(self) -> torch.Tensor:
        return torch.tensor([0, 1])

    def test_learning_filter_parameter(
        self,
        learning_filter: LearningHmmFilter,
    ) -> None:
        assert learning_filter.state_dim == 3
        assert learning_filter.discrete_observation_dim == 2
        assert learning_filter.emission.matrix.shape == (3, 2)
        assert learning_filter.transition.matrix.shape == (3, 3)
        assert learning_filter.transition.initial_state.shape == (3,)

        compile_config = CompileConfig()
        compile_config.stance = CompileConfig.Stance.FORCE_EAGER
        with CompileContext(compile_config):
            with BaseLearningProcess.inference_mode(learning_filter):
                assert_allclose(
                    learning_filter.transition.initial_state,
                    np.array([1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0]),
                )
                assert_allclose(
                    learning_filter.transition.matrix,
                    np.array(
                        [
                            [0.75, 0.25, 0.0],
                            [0.0, 0.75, 0.25],
                            [0.25, 0.0, 0.75],
                        ]
                    ),
                )
                assert_allclose(
                    learning_filter.emission.matrix,
                    np.array([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]]),
                )

    def test_learning_filter_estimation(
        self,
        learning_filter: LearningHmmFilter,
        observation_trajectory: torch.Tensor,
    ) -> None:

        compile_config = CompileConfig()
        compile_config.stance = CompileConfig.Stance.FORCE_EAGER
        with CompileContext(compile_config):
            with BaseLearningProcess.inference_mode(learning_filter):

                learning_filter.reset()
                assert_allclose(
                    learning_filter.transition.estimated_state,
                    np.array([1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0]),
                )

                learning_filter.update(observation_trajectory[0:1])
                assert_allclose(
                    learning_filter.transition.estimated_state,
                    np.array([0.425, 0.175, 0.4]),
                )

                learning_filter.update(observation_trajectory[1:2])
                assert_allclose(
                    learning_filter.transition.estimated_state,
                    np.array([22.75 / 85.0, 25.25 / 85.0, 37.0 / 85.0]),
                )
