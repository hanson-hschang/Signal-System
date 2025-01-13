import numpy as np
import pytest

from ss.system.markov import HiddenMarkovModel


class TestMarkovChain:
    @pytest.fixture
    def hidden_markov_model(self) -> HiddenMarkovModel:
        return HiddenMarkovModel(
            transition_probability_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]),
            emission_probability_matrix=np.array([[0.5, 0.5], [0.5, 0.5]]),
            initial_distribution=np.array([1.0, 0.0]),
            number_of_systems=2,
        )

    def test_initialization(
        self, hidden_markov_model: HiddenMarkovModel
    ) -> None:
        assert hidden_markov_model.discrete_state_dim == 2
        assert hidden_markov_model.discrete_observation_dim == 2
        assert hidden_markov_model.number_of_systems == 2

    def test_process(self, hidden_markov_model: HiddenMarkovModel) -> None:
        time = hidden_markov_model.process(0)
        assert time == 1
        np.testing.assert_allclose(hidden_markov_model.state, [[1], [1]])
        np.testing.assert_allclose(
            hidden_markov_model.state_one_hot, [[0.0, 1.0], [0.0, 1.0]]
        )
