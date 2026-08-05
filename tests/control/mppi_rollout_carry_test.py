import jax
import jax.numpy as jnp

from ss.control.mppi import MPPIController, RolloutCarry
from ss.system import CartPoleSystem


class TestRolloutCarry:
    def test_is_a_jittable_pytree(self) -> None:
        carry = RolloutCarry(
            times=jnp.zeros(2),
            states=jnp.ones((2, 3, 4)),
            costs=jnp.zeros((2, 3)),
        )

        @jax.jit
        def advance(carry: RolloutCarry) -> RolloutCarry:
            return RolloutCarry(
                times=carry.times + 0.1,
                states=carry.states + 1,
                costs=carry.costs + 2,
            )

        next_carry = advance(carry)

        assert jnp.allclose(next_carry.times, jnp.full(2, 0.1))
        assert jnp.array_equal(next_carry.states, jnp.full((2, 3, 4), 2))
        assert jnp.array_equal(next_carry.costs, jnp.full((2, 3), 2))

    def test_mppi_rollout_uses_structured_carry(self) -> None:
        system = CartPoleSystem(time_step=0.01, batch_size=2)
        controller = MPPIController(
            control_dim=system.control_dim,
            batch_size=system.batch_size,
            rollout_system=system,
            running_cost=lambda state, control: jnp.sum(state**2, axis=-1)
            + jnp.sum(control**2, axis=-1),
            terminal_cost=lambda state: jnp.sum(state**2, axis=-1),
            horizon=2,
            num_rollouts=4,
        )

        control, next_state, diagnostics = controller(
            controller.initial_state(),
            jnp.array(0.0),
            system.initial_state(),
            jax.random.PRNGKey(0),
        )

        assert control.shape == (system.batch_size, system.control_dim)
        assert next_state.nominal_controls.shape == (system.batch_size, 2, system.control_dim)
        assert diagnostics.minimum_rollout_cost.shape == (system.batch_size,)
        assert jnp.all(jnp.isfinite(control))
