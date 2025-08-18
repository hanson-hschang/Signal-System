import numpy as np
import pytest

from ss.estimation.filtering.hmm import HmmFilter
from ss.system.markov import HiddenMarkovModel


class TestHmmFilter:
    @pytest.fixture
    def hmm(self) -> HiddenMarkovModel:
        return HiddenMarkovModel(
            transition_matrix=np.array(
                [[0.75, 0.25, 0.0], [0.0, 0.75, 0.25], [0.25, 0.0, 0.75]]
            ),
            emission_matrix=np.array([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]]),
        )

    @pytest.fixture
    def filter(self, hmm: HiddenMarkovModel) -> HmmFilter:
        return HmmFilter(system=hmm)

    def test_hmm_filter(self, filter: HmmFilter) -> None:
        # Set estimated distribution of initial state
        filter.estimated_state = np.array([1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0])
        assert np.allclose(
            filter.estimated_state, np.array([1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0])
        )

        # First update
        filter.update(observation=np.array([0]))
        assert np.allclose(
            filter.estimated_state,
            np.array([0.425, 0.175, 0.4]),
            atol=1e-7,
        )

        # Second update
        filter.update(observation=np.array([1]))
        assert np.allclose(
            filter.estimated_state,
            np.array([22.75 / 85.0, 25.25 / 85.0, 37.0 / 85.0]),
            atol=1e-7,
        )
