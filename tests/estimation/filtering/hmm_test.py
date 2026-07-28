import pytest

import jax
import jax.numpy as jnp

from ss.estimation.filtering import HmmFilter, filtering
from ss.system import HiddenMarkovModel, simulate
from ss.utility.parameter.probability import ProbabilityParameter


class TestHmmFilter:
    @pytest.fixture
    def hmm_filter(self) -> HmmFilter:
        transition = jnp.array(
            [[0.75, 0.25, 0.0], [0.0, 0.75, 0.25], [0.25, 0.0, 0.75]]
        )
        emission = jnp.array([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]])
        return HmmFilter(
            transition=ProbabilityParameter(transition),
            emission=ProbabilityParameter(emission),
            state_dim=3,
            batch_size=1,
        )

    def test_hmm_filter_dimensions(self, hmm_filter: HmmFilter) -> None:
        assert hmm_filter.state_dim == 3
        assert hmm_filter.observation_dim == 1
        assert hmm_filter.control_dim == 0
        assert hmm_filter.batch_size == 1
        assert hmm_filter.time_step == 1.0
        assert hmm_filter.transition_matrix.shape == (3, 3)
        assert hmm_filter.emission_matrix.shape == (3, 2)

    def test_hmm_filter_update_and_estimate(
        self, hmm_filter: HmmFilter
    ) -> None:
        prior = jnp.array([[1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0]])

        posterior = hmm_filter.update(0.0, prior, jnp.array([[0]]))
        assert jnp.allclose(posterior, jnp.array([[0.4, 0.1, 0.5]]), atol=1e-7)

        estimated = hmm_filter.estimate(0.0, posterior)
        assert jnp.allclose(
            estimated, jnp.array([[0.425, 0.175, 0.4]]), atol=1e-7
        )

        posterior = hmm_filter.update(1.0, estimated, jnp.array([[1]]))
        assert jnp.allclose(
            posterior,
            jnp.array([[0.2, 0.3294117647, 0.4705882353]]),
            atol=1e-7,
        )

    def test_hmm_filter_with_matrices_is_immutable(
        self, hmm_filter: HmmFilter
    ) -> None:
        new_transition = jnp.eye(3)
        new_emission = jnp.ones((3, 2)) / 2.0

        updated = hmm_filter.with_transition_matrix(
            new_transition
        ).with_emission_matrix(new_emission)

        assert updated is not hmm_filter
        assert jnp.allclose(updated.transition_matrix, new_transition)
        assert jnp.allclose(updated.emission_matrix, new_emission)
        assert not jnp.allclose(hmm_filter.transition_matrix, new_transition)

    def test_filtering_single_and_batch(self, hmm_filter: HmmFilter) -> None:
        initial_belief = jnp.array([[1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0]])
        observations = jnp.array([[[0]], [[1]], [[0]]])  # (time, batch, obs)

        times, beliefs = filtering(hmm_filter, 0.0, initial_belief, observations)
        assert times.shape == (3,)
        assert jnp.allclose(times, jnp.array([1.0, 2.0, 3.0]))
        assert beliefs.shape == (3, 1, 3)
        assert jnp.allclose(
            beliefs[0, 0], jnp.array([0.4, 0.1, 0.5]), atol=1e-7
        )

        hmm_filter = hmm_filter.duplicate(batch_size=2)
        batch_observations = jnp.concatenate(
            [observations, observations], axis=1
        )
        initial_beliefs = jnp.concatenate(
            [initial_belief, initial_belief], axis=0
        )
        batch_times, batch_beliefs = filtering(
            hmm_filter, 0.0, initial_beliefs, batch_observations
        )
        assert batch_times.shape == (3,)
        assert batch_beliefs.shape == (3, 2, 3)
        assert jnp.allclose(batch_beliefs[:, 0], beliefs[:, 0])
        assert jnp.allclose(batch_beliefs[:, 1], beliefs[:, 0])

    def test_filtering_is_jittable(self, hmm_filter: HmmFilter) -> None:
        initial_belief = jnp.array([[1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0]])
        observations = jnp.array([[[0]], [[1]]])  # (time, batch, obs)

        jitted = jax.jit(filtering)
        times, beliefs = jitted(hmm_filter, 0.0, initial_belief, observations)
        assert times.shape == (2,)
        assert beliefs.shape == (2, 1, 3)

        hmm_filter = hmm_filter.duplicate(batch_size=2)
        batch_times, batch_beliefs = jitted(
            hmm_filter,
            0.0,
            jnp.concatenate([initial_belief, initial_belief], axis=0),
            jnp.concatenate([observations, observations], axis=1),
        )
        assert batch_times.shape == (2,)
        assert batch_beliefs.shape == (2, 2, 3)

    def test_filtering_from_simulate_batch(self) -> None:
        """simulate/filtering share time-leading layout; state aligns with obs."""
        transition = jnp.array([[0.7, 0.3], [0.4, 0.6]])
        emission = jnp.array([[0.9, 0.1], [0.2, 0.8]])
        system = HiddenMarkovModel(
            transition=ProbabilityParameter(transition),
            emission=ProbabilityParameter(emission),
        ).duplicate(batch_size=4)
        hmm_filter = HmmFilter(
            transition=ProbabilityParameter(transition),
            emission=ProbabilityParameter(emission),
            state_dim=2,
            batch_size=4,
        )

        key = jax.random.PRNGKey(0)
        init_key, scan_key = jax.random.split(key)
        init_states = system.initial_state(init_key)
        sim_times, states, observations, _ = simulate(
            system, 0.0, 6, init_states, scan_key
        )
        assert states.shape[:2] == observations.shape[:2] == (6, 4)
        assert observations.shape == (6, 4, 1)

        initial_belief = jnp.full((4, 2), 0.5)
        filter_times, beliefs = filtering(
            hmm_filter, 0.0, initial_belief, observations
        )
        assert jnp.allclose(filter_times, sim_times)
        assert beliefs.shape == (6, 4, 2)
        assert jnp.allclose(jnp.sum(beliefs, axis=-1), 1.0)
