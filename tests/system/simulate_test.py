import jax
import jax.numpy as jnp

from ss.system import CartPoleSystem, SimulationResult, simulate


class TestSimulation:
    def test_includes_aligned_initial_and_terminal_states(self) -> None:
        system = CartPoleSystem(time_step=0.1, batch_size=2)
        initial_state = jnp.array([[0.2, 0.0, 0.1, 0.0], [-0.3, 0.0, -0.2, 0.0]])

        result = simulate(
            system,
            1.0,
            3,
            initial_state,
            jax.random.PRNGKey(0),
        )

        assert isinstance(result, SimulationResult)
        assert result.times.shape == (4,)
        assert result.states.shape == (4, 2, system.state_dim)
        assert result.observations.shape == (3, 2, system.observation_dim)
        assert result.controls is None
        assert result.controller_diagnostics is None
        assert jnp.array_equal(result.states[0], initial_state)
        assert jnp.allclose(result.times, jnp.array([1.0, 1.1, 1.2, 1.3]))
        assert jnp.array_equal(result.observations[0], initial_state)
