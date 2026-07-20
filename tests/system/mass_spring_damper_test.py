import equinox as eqx
import jax
import jax.numpy as jnp

from ss.system import (
    ControlChoice,
    MassSpringDamperSystem,
    ObservationChoice,
)


def test_mass_spring_damper_builds_expected_continuous_dynamics() -> None:
    system = MassSpringDamperSystem(number_of_connections=2)

    expected_a = jnp.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-2.0, 1.0, -2.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
        ]
    )
    assert jnp.allclose(system.continuous_state_matrix, expected_a)


def test_observation_and_control_choices_select_physical_coordinates() -> None:
    position_system = MassSpringDamperSystem(
        number_of_connections=2,
        observation_choice=ObservationChoice.ALL_POSITIONS,
        control_choice=ControlChoice.LAST_FORCE,
    )
    state = jnp.array([2.0, 3.0, 5.0, 7.0])

    observation = position_system.observe(0.0, state, jax.random.PRNGKey(0))

    assert jnp.array_equal(observation, jnp.array([2.0, 3.0]))
    assert position_system.observation_dim == 2
    assert position_system.control_dim == 1
    assert position_system.continuous_control_matrix[-1, 0] == 1.0

    last_position_system = MassSpringDamperSystem(
        number_of_connections=2,
        observation_choice=ObservationChoice.LAST_POSITION,
    )
    assert jnp.array_equal(
        last_position_system.observe(0.0, state, jax.random.PRNGKey(1)),
        jnp.array([3.0]),
    )


def test_exact_discrete_step_matches_exposed_state_space_matrices() -> None:
    system = MassSpringDamperSystem(
        number_of_connections=2,
        time_step=0.05,
        control_choice=ControlChoice.ALL_FORCES,
    )
    state = jnp.array([0.2, -0.1, 0.3, 0.4])
    control = jnp.array([1.0, -2.0])

    next_time, next_state = eqx.filter_jit(system.process)(
        0.0,
        state,
        jax.random.PRNGKey(0),
        control=control,
    )

    expected = (
        system.discrete_state_matrix @ state
        + system.discrete_control_matrix @ control
    )
    assert jnp.allclose(next_time, system.time_step)
    assert jnp.allclose(next_state, expected)


def test_mass_spring_damper_initialization_and_duplication_are_batched() -> (
    None
):
    system = MassSpringDamperSystem(
        number_of_connections=3,
        batch_size=2,
    )

    initial_state = system.init_state(jax.random.PRNGKey(0))
    duplicate = system.duplicate(batch_size=5)

    assert initial_state.shape == (2, 6)
    assert duplicate.init_state().shape == (5, 6)
    assert duplicate.discrete_state_matrix.shape == (6, 6)
    assert system.batch_size == 2
