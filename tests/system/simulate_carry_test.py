import jax
import jax.numpy as jnp

from ss.system._system import SimulateCarry


def test_simulate_carry_is_a_jittable_pytree() -> None:
    carry = SimulateCarry(
        time=0.0,
        state=jnp.ones((2, 3)),
        controller_state={"step": jnp.array(0)},
    )

    @jax.jit
    def advance(carry: SimulateCarry) -> SimulateCarry:
        return SimulateCarry(
            time=carry.time + 0.1,
            state=carry.state + 1,
            controller_state={"step": carry.controller_state["step"] + 1},
        )

    next_carry = advance(carry)

    assert jnp.allclose(next_carry.time, 0.1)
    assert jnp.array_equal(next_carry.state, jnp.full((2, 3), 2))
    assert next_carry.controller_state["step"] == 1


def test_simulate_carry_supports_no_controller_state() -> None:
    carry = SimulateCarry(
        time=0.0,
        state=jnp.ones((1, 2)),
        controller_state=None,
    )

    leaves = jax.tree.leaves(carry)

    assert len(leaves) == 2
