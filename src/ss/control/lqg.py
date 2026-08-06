# ruff: noqa: F722, F821

from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ._control import Controller


class LinearStateSpaceSystem(Protocol):
    """Structural interface required by ``LQGController.from_system``."""

    state_dim: int
    observation_dim: int
    control_dim: int
    batch_size: int
    discrete_state_matrix: Array
    discrete_control_matrix: Array
    observation_matrix: Array
    process_noise_covariance: Array
    observation_noise_covariance: Array


class LQGControllerState(eqx.Module):
    predicted_state: Float[Array, "batch_size state_dim"]


class LQGDiagnostics(eqx.Module):
    estimated_state: Float[Array, "batch_size state_dim"]
    innovation: Float[Array, "batch_size observation_dim"]


class LQGController(Controller):
    """Infinite-horizon discrete LQG controller with a steady-state filter."""

    state_dim: int = eqx.field(static=True)
    observation_dim: int = eqx.field(static=True)
    riccati_iterations: int = eqx.field(static=True)

    state_matrix: Float[Array, "state_dim state_dim"]
    control_matrix: Float[Array, "state_dim control_dim"]
    observation_matrix: Float[Array, "observation_dim state_dim"]
    state_cost: Float[Array, "state_dim state_dim"]
    control_cost: Float[Array, "control_dim control_dim"]
    process_noise_covariance: Float[Array, "state_dim state_dim"]
    observation_noise_covariance: Float[Array, "observation_dim observation_dim"]
    feedback_gain: Float[Array, "control_dim state_dim"]
    estimator_gain: Float[Array, "state_dim observation_dim"]

    def __init__(
        self,
        *,
        control_dim: int,
        batch_size: int,
        state_matrix: Array,
        control_matrix: Array,
        observation_matrix: Array,
        state_cost: Array,
        control_cost: Array,
        process_noise_covariance: Array,
        observation_noise_covariance: Array,
        riccati_iterations: int = 500,
    ) -> None:
        state_matrix = jnp.asarray(state_matrix)
        control_matrix = jnp.asarray(control_matrix)
        observation_matrix = jnp.asarray(observation_matrix)
        state_cost = jnp.asarray(state_cost)
        control_cost = jnp.asarray(control_cost)
        process_noise_covariance = jnp.asarray(process_noise_covariance)
        observation_noise_covariance = jnp.asarray(observation_noise_covariance)
        state_dim = state_matrix.shape[0]
        observation_dim = observation_matrix.shape[0]
        self._validate_configuration(
            control_dim,
            batch_size,
            state_dim,
            observation_dim,
            state_matrix,
            control_matrix,
            observation_matrix,
            state_cost,
            control_cost,
            process_noise_covariance,
            observation_noise_covariance,
            riccati_iterations,
        )

        self.control_dim = control_dim
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.riccati_iterations = riccati_iterations
        self.state_matrix = state_matrix
        self.control_matrix = control_matrix
        self.observation_matrix = observation_matrix
        self.state_cost = state_cost
        self.control_cost = control_cost
        self.process_noise_covariance = process_noise_covariance
        self.observation_noise_covariance = observation_noise_covariance
        self.feedback_gain = self._solve_feedback_gain()
        self.estimator_gain = self._solve_estimator_gain()

    def __check_init__(self) -> None:
        super().__check_init__()
        assert self.state_dim > 0
        assert self.observation_dim > 0
        assert self.riccati_iterations > 0
        assert self.state_matrix.shape == (self.state_dim, self.state_dim)
        assert self.control_matrix.shape == (
            self.state_dim,
            self.control_dim,
        )
        assert self.observation_matrix.shape == (
            self.observation_dim,
            self.state_dim,
        )
        assert self.state_cost.shape == (self.state_dim, self.state_dim)
        assert self.control_cost.shape == (
            self.control_dim,
            self.control_dim,
        )
        assert self.process_noise_covariance.shape == (
            self.state_dim,
            self.state_dim,
        )
        assert self.observation_noise_covariance.shape == (
            self.observation_dim,
            self.observation_dim,
        )

    @classmethod
    def from_system(
        cls,
        system: LinearStateSpaceSystem,
        *,
        state_cost: Array,
        control_cost: Array,
        process_noise_covariance: Array | None = None,
        observation_noise_covariance: Array | None = None,
        riccati_iterations: int = 500,
    ) -> "LQGController":
        if process_noise_covariance is None:
            process_noise_covariance = system.process_noise_covariance
        if observation_noise_covariance is None:
            observation_noise_covariance = system.observation_noise_covariance
        return cls(
            control_dim=system.control_dim,
            batch_size=system.batch_size,
            state_matrix=system.discrete_state_matrix,
            control_matrix=system.discrete_control_matrix,
            observation_matrix=system.observation_matrix,
            state_cost=state_cost,
            control_cost=control_cost,
            process_noise_covariance=process_noise_covariance,
            observation_noise_covariance=observation_noise_covariance,
            riccati_iterations=riccati_iterations,
        )

    @staticmethod
    def _validate_configuration(
        control_dim: int,
        batch_size: int,
        state_dim: int,
        observation_dim: int,
        state_matrix: Array,
        control_matrix: Array,
        observation_matrix: Array,
        state_cost: Array,
        control_cost: Array,
        process_noise_covariance: Array,
        observation_noise_covariance: Array,
        riccati_iterations: int,
    ) -> None:
        assert control_dim > 0
        assert batch_size > 0
        assert state_dim > 0
        assert observation_dim > 0
        assert riccati_iterations > 0
        assert state_matrix.shape == (state_dim, state_dim)
        assert control_matrix.shape == (state_dim, control_dim)
        assert observation_matrix.shape == (observation_dim, state_dim)
        assert state_cost.shape == (state_dim, state_dim)
        assert control_cost.shape == (control_dim, control_dim)
        assert process_noise_covariance.shape == (state_dim, state_dim)
        assert observation_noise_covariance.shape == (
            observation_dim,
            observation_dim,
        )

        for name, matrix in (
            ("state_cost", state_cost),
            ("control_cost", control_cost),
            ("process_noise_covariance", process_noise_covariance),
            ("observation_noise_covariance", observation_noise_covariance),
        ):
            assert jnp.allclose(matrix, matrix.T), f"{name} must be symmetric"
        assert jnp.all(jnp.linalg.eigvalsh(state_cost) >= 0), "state_cost must be positive semidefinite"
        assert jnp.all(jnp.linalg.eigvalsh(control_cost) > 0), "control_cost must be positive definite"
        assert jnp.all(jnp.linalg.eigvalsh(process_noise_covariance) >= 0), (
            "process_noise_covariance must be positive semidefinite"
        )
        assert jnp.all(jnp.linalg.eigvalsh(observation_noise_covariance) > 0), (
            "observation_noise_covariance must be positive definite"
        )

    def initial_state(
        self,
        random_key: PRNGKeyArray | None = None,
    ) -> LQGControllerState:
        del random_key
        return LQGControllerState(jnp.zeros((self.batch_size, self.state_dim)))

    def __call__(
        self,
        controller_state: LQGControllerState,
        time: Float[Array, ""],
        observation: Float[Array, "batch_size observation_dim"],
        random_key: PRNGKeyArray,
    ) -> tuple[
        Float[Array, "batch_size control_dim"],  # control
        LQGControllerState,  # next controller state
        LQGDiagnostics,  # diagnostics
    ]:
        del time, random_key
        predicted_observation = controller_state.predicted_state @ self.observation_matrix.T
        innovation = observation - predicted_observation
        estimated_state = controller_state.predicted_state + innovation @ self.estimator_gain.T
        control = -estimated_state @ self.feedback_gain.T
        predicted_state = estimated_state @ self.state_matrix.T + control @ self.control_matrix.T
        return (
            control,
            LQGControllerState(predicted_state),
            LQGDiagnostics(estimated_state, innovation),
        )

    def _solve_feedback_gain(self) -> Array:
        a = self.state_matrix
        b = self.control_matrix
        q = self.state_cost
        r = self.control_cost

        def riccati_step(_: int, covariance: Array) -> Array:
            gain = jnp.linalg.solve(
                r + b.T @ covariance @ b,
                b.T @ covariance @ a,
            )
            return q + a.T @ covariance @ a - a.T @ covariance @ b @ gain

        covariance = jax.lax.fori_loop(0, self.riccati_iterations, riccati_step, q)
        return jnp.linalg.solve(
            r + b.T @ covariance @ b,
            b.T @ covariance @ a,
        )

    def _solve_estimator_gain(self) -> Array:
        a = self.state_matrix
        c = self.observation_matrix
        process_covariance = self.process_noise_covariance
        observation_covariance = self.observation_noise_covariance

        def riccati_step(_: int, predicted_covariance: Array) -> Array:
            innovation_covariance = c @ predicted_covariance @ c.T + observation_covariance
            gain = jnp.linalg.solve(
                innovation_covariance,
                c @ predicted_covariance,
            ).T
            corrected_covariance = predicted_covariance - gain @ c @ predicted_covariance
            return a @ corrected_covariance @ a.T + process_covariance

        predicted_covariance = jax.lax.fori_loop(
            0,
            self.riccati_iterations,
            riccati_step,
            process_covariance,
        )
        innovation_covariance = c @ predicted_covariance @ c.T + observation_covariance
        return jnp.linalg.solve(
            innovation_covariance,
            c @ predicted_covariance,
        ).T
