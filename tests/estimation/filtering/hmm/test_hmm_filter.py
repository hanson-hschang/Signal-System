import numpy as np
import pytest
from numpy.typing import NDArray

from ss.estimation.filtering.hmm import DualHmmFilter, HmmFilter
from ss.system.markov import HiddenMarkovModel


class TestHmmFilter:
    @pytest.fixture
    def initial_distribution(self) -> NDArray:
        return np.array([1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0])

    @pytest.fixture
    def hmm(self) -> HiddenMarkovModel:
        return HiddenMarkovModel(
            transition_matrix=np.array(
                [[0.75, 0.25, 0.0], [0.0, 0.75, 0.25], [0.25, 0.0, 0.75]]
            ),
            emission_matrix=np.array([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]]),
        )

    @pytest.fixture
    def filter(
        self, hmm: HiddenMarkovModel, initial_distribution: NDArray
    ) -> HmmFilter:
        return HmmFilter(system=hmm, initial_distribution=initial_distribution)

    @pytest.fixture
    def dual_filter(
        self, hmm: HiddenMarkovModel, initial_distribution: NDArray
    ) -> DualHmmFilter:
        return DualHmmFilter(
            system=hmm,
            horizon_of_observation_history=10,
            initial_distribution=initial_distribution,
        )

    def test_hmm_filter(self, filter: HmmFilter) -> None:
        # Test estimated distribution of initial state
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

    def test_dual_hmm_filter(self, dual_filter: DualHmmFilter) -> None:
        # Test estimated distribution of initial state
        assert np.allclose(
            dual_filter.estimated_state,
            np.array([1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0]),
        )

        # First update
        dual_filter.update(observation=np.array([0]))
        assert np.allclose(
            dual_filter.estimated_state,
            np.array([0.425, 0.175, 0.4]),
            atol=1e-7,
        )

        # Second update
        dual_filter.update(observation=np.array([1]))
        assert np.allclose(
            dual_filter.estimated_state,
            np.array([22.75 / 85.0, 25.25 / 85.0, 37.0 / 85.0]),
            atol=1e-7,
        )
