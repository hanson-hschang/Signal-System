import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from ss.control import LQGController, LQGControllerState
from ss.system import (
    ControlChoice,
    MassSpringDamperSystem,
    ObservationChoice,
    simulate,
)


class TestLQGController:
    def test_computes_the_scalar_infinite_horizon_lqr_gain(self) -> None:
        controller = LQGController(
            control_dim=1,
            batch_size=1,
            state_matrix=jnp.array([[1.0]]),
            control_matrix=jnp.array([[1.0]]),
            observation_matrix=jnp.array([[1.0]]),
            state_cost=jnp.array([[1.0]]),
            control_cost=jnp.array([[1.0]]),
            process_noise_covariance=jnp.array([[0.1]]),
            observation_noise_covariance=jnp.array([[0.2]]),
        )

        golden_ratio_conjugate = (jnp.sqrt(5.0) - 1.0) / 2.0
        assert jnp.allclose(controller.feedback_gain[0, 0], golden_ratio_conjugate, atol=1e-5)

    def test_updates_its_estimate_and_returns_named_diagnostics(self) -> None:
        controller = LQGController(
            control_dim=1,
            batch_size=2,
            state_matrix=jnp.array([[1.0]]),
            control_matrix=jnp.array([[1.0]]),
            observation_matrix=jnp.array([[1.0]]),
            state_cost=jnp.array([[1.0]]),
            control_cost=jnp.array([[1.0]]),
            process_noise_covariance=jnp.array([[0.1]]),
            observation_noise_covariance=jnp.array([[0.2]]),
        )
        state = controller.initial_state()
        observation = jnp.array([[2.0], [-1.0]])

        control, next_state, diagnostics = eqx.filter_jit(controller)(
            state,
            jnp.array(0.0),
            observation,
            jax.random.PRNGKey(0),
        )

        expected_estimate = observation * controller.estimator_gain[0, 0]
        assert isinstance(next_state, LQGControllerState)
        assert jnp.allclose(diagnostics.estimated_state, expected_estimate)
        assert jnp.array_equal(diagnostics.innovation, observation)
        assert jnp.allclose(control, -expected_estimate @ controller.feedback_gain.T)
        assert jnp.allclose(
            next_state.predicted_state,
            expected_estimate @ controller.state_matrix.T + control @ controller.control_matrix.T,
        )

    def test_validates_weights_before_solving_riccati_equations(self) -> None:
        with pytest.raises(AssertionError, match="control_cost must be positive"):
            LQGController(
                control_dim=1,
                batch_size=1,
                state_matrix=jnp.eye(1),
                control_matrix=jnp.ones((1, 1)),
                observation_matrix=jnp.eye(1),
                state_cost=jnp.eye(1),
                control_cost=jnp.zeros((1, 1)),
                process_noise_covariance=jnp.eye(1),
                observation_noise_covariance=jnp.eye(1),
            )

    def test_controls_batched_partially_observed_systems(self) -> None:
        system = MassSpringDamperSystem(
            number_of_connections=2,
            time_step=0.02,
            observation_choice=ObservationChoice.ALL_POSITIONS,
            control_choice=ControlChoice.ALL_FORCES,
            batch_size=3,
        )
        controller = LQGController.from_system(
            system,
            state_cost=jnp.eye(system.state_dim),
            control_cost=0.1 * jnp.eye(system.control_dim),
            process_noise_covariance=0.01 * jnp.eye(system.state_dim),
            observation_noise_covariance=0.02 * jnp.eye(system.observation_dim),
        )
        initial_state = jnp.array(
            [
                [1.0, -0.5, 0.0, 0.0],
                [-0.5, 0.2, 0.1, 0.0],
                [0.3, 0.8, 0.0, -0.1],
            ]
        )

        result = simulate(
            system,
            0.0,
            8,
            initial_state,
            jax.random.PRNGKey(0),
            controller,
        )

        assert result.times.shape == (9,)
        assert result.states.shape == (9, 3, 4)
        assert result.observations.shape == (8, 3, 2)
        assert result.controls.shape == (8, 3, 2)
        assert result.controller_diagnostics.estimated_state.shape == (8, 3, 4)
        assert jnp.array_equal(result.states[0], initial_state)
        assert jnp.all(jnp.isfinite(result.states))
        assert jnp.all(jnp.isfinite(result.controls))
