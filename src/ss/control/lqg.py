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
    discrete_state_matrix: Float[Array, "state_dim state_dim"]
    discrete_control_matrix: Float[Array, "state_dim control_dim"]
    observation_matrix: Float[Array, "observation_dim state_dim"]
    process_noise_covariance: Float[Array, "state_dim state_dim"]
    observation_noise_covariance: Float[Array, "observation_dim observation_dim"]


class LQGControllerState(eqx.Module):
    predicted_state: Float[Array, "batch_size state_dim"]


class LQGDiagnostics(eqx.Module):
    estimated_state: Float[Array, "batch_size state_dim"]
    innovation: Float[Array, "batch_size observation_dim"]


class LQGController(Controller):
    """Infinite-horizon discrete LQG controller with a steady-state filter."""

    state_matrix: Float[Array, "state_dim state_dim"] = eqx.field(converter=jnp.asarray)
    control_matrix: Float[Array, "state_dim control_dim"] = eqx.field(converter=jnp.asarray)
    observation_matrix: Float[Array, "observation_dim state_dim"] = eqx.field(converter=jnp.asarray)
    state_cost: Float[Array, "state_dim state_dim"] = eqx.field(converter=jnp.asarray)
    control_cost: Float[Array, "control_dim control_dim"] = eqx.field(converter=jnp.asarray)
    process_noise_covariance: Float[Array, "state_dim state_dim"] = eqx.field(converter=jnp.asarray)
    observation_noise_covariance: Float[Array, "observation_dim observation_dim"] = eqx.field(converter=jnp.asarray)

    state_dim: int = eqx.field(static=True, default=0)
    observation_dim: int = eqx.field(static=True, default=0)
    riccati_iterations: int = eqx.field(static=True, default=500)
    feedback_gain: Float[Array, "control_dim state_dim"] = eqx.field(
        default=0.0,
        converter=jnp.asarray,
    )
    estimator_gain: Float[Array, "state_dim observation_dim"] = eqx.field(
        default=0.0,
        converter=jnp.asarray,
    )

    def __post_init__(self) -> None:
        self.state_dim = int(self.state_matrix.shape[0])
        self.observation_dim = int(self.observation_matrix.shape[0])
        self.feedback_gain = self._solve_feedback_gain()
        self.estimator_gain = self._solve_estimator_gain()

    def __check_init__(self) -> None:
        super().__check_init__()
        assert self.state_dim > 0
        assert self.observation_dim > 0
        assert self.riccati_iterations > 0
        assert self.state_matrix.shape == (self.state_dim, self.state_dim)
        assert self.control_matrix.shape == (self.state_dim, self.control_dim)
        assert self.observation_matrix.shape == (self.observation_dim, self.state_dim)
        assert self.state_cost.shape == (self.state_dim, self.state_dim)
        assert self.control_cost.shape == (self.control_dim, self.control_dim)
        assert self.process_noise_covariance.shape == (self.state_dim, self.state_dim)
        assert self.observation_noise_covariance.shape == (
            self.observation_dim,
            self.observation_dim,
        )
        assert self.feedback_gain.shape == (self.control_dim, self.state_dim)
        assert self.estimator_gain.shape == (self.state_dim, self.observation_dim)

        for name, matrix in (
            ("state_cost", self.state_cost),
            ("control_cost", self.control_cost),
            ("process_noise_covariance", self.process_noise_covariance),
            ("observation_noise_covariance", self.observation_noise_covariance),
        ):
            assert jnp.allclose(matrix, matrix.T), f"{name} must be symmetric"
        assert jnp.all(jnp.linalg.eigvalsh(self.state_cost) >= 0), "state_cost must be positive semidefinite"
        assert jnp.all(jnp.linalg.eigvalsh(self.control_cost) > 0), "control_cost must be positive definite"
        assert jnp.all(jnp.linalg.eigvalsh(self.process_noise_covariance) >= 0), (
            "process_noise_covariance must be positive semidefinite"
        )
        assert jnp.all(jnp.linalg.eigvalsh(self.observation_noise_covariance) > 0), (
            "observation_noise_covariance must be positive definite"
        )

    @classmethod
    def from_system(
        cls,
        system: LinearStateSpaceSystem,
        *,
        state_cost: Float[Array, "state_dim state_dim"],
        control_cost: Float[Array, "control_dim control_dim"],
        process_noise_covariance: Float[Array, "state_dim state_dim"] | None = None,
        observation_noise_covariance: Float[Array, "observation_dim observation_dim"] | None = None,
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

    def initial_state(
        self,
        random_key: PRNGKeyArray | None = None,
    ) -> LQGControllerState:
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

    def _solve_feedback_gain(self) -> Float[Array, "control_dim state_dim"]:
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

    def _solve_estimator_gain(self) -> Float[Array, "state_dim observation_dim"]:
        a = self.state_matrix
        c = self.observation_matrix
        process_covariance = self.process_noise_covariance
        observation_covariance = self.observation_noise_covariance

        def riccati_step(
            _: int, predicted_covariance: Float[Array, "state_dim state_dim"]
        ) -> Float[Array, "state_dim state_dim"]:
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
