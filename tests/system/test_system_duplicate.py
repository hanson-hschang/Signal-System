import jax.numpy as jnp

from ss.system import CartPoleSystem


def test_duplicate_system_with_new_batch_size() -> None:
    system = CartPoleSystem(
        time_step=0.02,
        cart_mass=2.0,
        pole_mass=0.2,
        pole_length=1.5,
        gravity=9.7,
        batch_size=1,
    )

    duplicate = system.duplicate(batch_size=8)

    assert duplicate is not system
    assert system.batch_size == 1
    assert duplicate.batch_size == 8
    assert duplicate.time_step == system.time_step
    assert duplicate.cart_mass == system.cart_mass
    assert duplicate.pole_mass == system.pole_mass
    assert duplicate.pole_length == system.pole_length
    assert duplicate.gravity == system.gravity
    assert duplicate.init_state().shape == (8, system.state_dim)
    assert jnp.all(duplicate.init_state() == 0)
