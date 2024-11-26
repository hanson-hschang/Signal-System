import numpy as np
import pytest

from ss.system.state_vector.markov import MarkovChain


class TestMarkovChain:

    @pytest.fixture
    def markov_chain(self) -> MarkovChain:
        """Create a basic markov_chain with default parameters"""
        return MarkovChain(
            transition_probability_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]),
            emission_probability_matrix=np.array([[0.5, 0.5], [0.5, 0.5]]),
            initial_distribution=np.array([0.5, 0.5]),
            number_of_systems=10,
        )

    def test_initialization(self, markov_chain: MarkovChain) -> None:
        assert markov_chain.state_dim == 2
        assert markov_chain.observation_dim == 2
        assert markov_chain.number_of_systems == 10

    def test_process(self, markov_chain: MarkovChain) -> None:
        state = markov_chain.state
        time = markov_chain.process(0)
        assert time == 1
        np.testing.assert_allclose(markov_chain.state, state[:, ::-1])
