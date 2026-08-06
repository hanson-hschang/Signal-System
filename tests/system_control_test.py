import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, PRNGKeyArray

from ss.control import Controller, MPPIController
from ss.system import CartPoleSystem, simulate


class ControllerStep(eqx.Module):
    value: Array


class FeedbackController(Controller):
    """Minimal example of the stateful controller protocol."""

    gain: float

    def initial_state(
        self,
        random_key: PRNGKeyArray | None = None,
    ) -> ControllerStep:
        del random_key
        return ControllerStep(jnp.array(0))

    def __call__(
        self,
        controller_state: ControllerStep,
        time: Array,
        observation: Array,
        random_key: PRNGKeyArray,
    ) -> tuple[
        Array,  # control
        ControllerStep,  # next controller state
        tuple[()],  # diagnostics
    ]:
        del time, random_key
        control = -self.gain * observation[:, :1] + controller_state.value
        next_state = ControllerStep(controller_state.value + 1)
        return control, next_state, ()


class QuadraticCost(eqx.Module):
    """Small cost model used to exercise model-based control."""

    def running_cost(self, state: Array, control: Array) -> Array:
        return jnp.sum(state**2, axis=-1) + 0.01 * jnp.sum(control**2, axis=-1)

    def terminal_cost(self, state: Array) -> Array:
        return 2 * jnp.sum(state**2, axis=-1)


class TestSystemControl:
    def test_uncontrolled_simulation_is_time_first_and_batched(self) -> None:
        system = CartPoleSystem(time_step=0.01, batch_size=3)
        initial_state = system.initial_state(jax.random.PRNGKey(0))

        result = simulate(
            system,
            0.0,
            4,
            initial_state,
            jax.random.PRNGKey(1),
        )

        assert result.times.shape == (5,)
        assert result.states.shape == (5, 3, system.state_dim)
        assert result.observations.shape == (4, 3, system.observation_dim)
        assert result.controls is None
        assert jnp.allclose(result.times, jnp.arange(5) * system.time_step)
        assert jnp.allclose(result.observations[0], initial_state)

    def test_simulation_accepts_a_stateful_controller(self) -> None:
        system = CartPoleSystem(time_step=0.01, batch_size=2)
        controller = FeedbackController(
            control_dim=system.control_dim,
            batch_size=system.batch_size,
            gain=0.5,
        )
        initial_state = jnp.array([[0.2, 0.0, 0.4, 0.0], [-0.1, 0.0, -0.3, 0.0]])

        result = simulate(
            system,
            0.0,
            3,
            initial_state,
            jax.random.PRNGKey(0),
            controller,
        )

        assert result.states.shape == (4, 2, system.state_dim)
        assert result.controls.shape == (3, 2, system.control_dim)
        assert jnp.allclose(
            result.controls[0],
            -0.5 * result.observations[0, :, :1],
        )
        assert jnp.allclose(
            result.controls[1],
            -0.5 * result.observations[1, :, :1] + 1,
        )

    def test_system_can_be_duplicated_for_model_rollouts(self) -> None:
        plant = CartPoleSystem(
            time_step=0.02,
            pole_mass=0.2,
            pole_length=1.5,
            batch_size=2,
        )

        rollout_system = plant.duplicate(batch_size=8)

        assert rollout_system is not plant
        assert plant.batch_size == 2
        assert rollout_system.batch_size == 8
        assert rollout_system.time_step == plant.time_step
        assert rollout_system.pole_mass == plant.pole_mass
        assert rollout_system.pole_length == plant.pole_length
        assert rollout_system.initial_state().shape == (8, plant.state_dim)

    @pytest.mark.parametrize(
        ("controller_batch_size", "controller_control_dim", "message"),
        [(3, 1, "batch_size"), (2, 2, "control_dim")],
    )
    def test_simulation_rejects_an_incompatible_controller(
        self,
        controller_batch_size: int,
        controller_control_dim: int,
        message: str,
    ) -> None:
        system = CartPoleSystem(batch_size=2)
        controller = FeedbackController(
            control_dim=controller_control_dim,
            batch_size=controller_batch_size,
            gain=0.5,
        )

        with pytest.raises(AssertionError, match=message):
            simulate(
                system,
                0.0,
                1,
                system.initial_state(),
                jax.random.PRNGKey(0),
                controller,
            )

    def test_mppi_controls_batched_systems_with_sampled_rollouts(self) -> None:
        system = CartPoleSystem(time_step=0.02, batch_size=2)
        cost = QuadraticCost()
        controller = MPPIController(
            control_dim=system.control_dim,
            batch_size=system.batch_size,
            rollout_system=system,
            running_cost=cost.running_cost,
            terminal_cost=cost.terminal_cost,
            horizon=3,
            num_rollouts=5,
        )
        initial_state = jnp.array([[0.0, 0.0, 0.4, 0.0], [0.0, 0.0, -0.3, 0.0]])

        result = simulate(
            system,
            0.0,
            2,
            initial_state,
            jax.random.PRNGKey(0),
            controller,
        )

        assert system.batch_size == 2
        assert controller.rollout_system.batch_size == 5
        assert controller.rollout_system is not system
        assert result.states.shape == (3, 2, system.state_dim)
        assert result.controls.shape == (2, 2, system.control_dim)
        assert jnp.all(jnp.isfinite(result.states))
        assert jnp.all(jnp.isfinite(result.controls))
