import equinox as eqx
import jax.numpy as jnp

from ss.control import QuadraticCost


class TestQuadraticCost:
    def test_running_and_terminal_costs_match_quadratic_forms(self) -> None:
        cost = QuadraticCost(
            state_weight=jnp.diag(jnp.array([2.0, 4.0])),
            control_weight=jnp.array([[6.0]]),
            terminal_scale=3.0,
        )
        state = jnp.array([[1.0, 0.5], [0.0, -1.0]])
        control = jnp.array([[2.0], [-0.5]])

        running = cost.running_cost(state, control)
        terminal = cost.terminal_cost(state)

        expected_state = 0.5 * jnp.array([2.0 * 1.0**2 + 4.0 * 0.5**2, 4.0 * 1.0**2])
        expected_control = 0.5 * jnp.array([6.0 * 2.0**2, 6.0 * 0.5**2])
        assert jnp.allclose(running, expected_state + expected_control)
        assert jnp.allclose(terminal, 3.0 * expected_state)

    def test_is_a_jittable_pytree(self) -> None:
        cost = QuadraticCost(
            state_weight=jnp.eye(2),
            control_weight=jnp.eye(1),
        )

        @eqx.filter_jit
        def evaluate(cost: QuadraticCost, state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
            return cost.running_cost(state, control) + cost.terminal_cost(state)

        values = evaluate(
            cost,
            jnp.array([[1.0, -1.0]]),
            jnp.array([[0.5]]),
        )
        assert values.shape == (1,)
