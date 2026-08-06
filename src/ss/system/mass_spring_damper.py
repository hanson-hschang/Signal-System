# ruff: noqa: F722, F821

from dataclasses import InitVar
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jaxtyping import Array, Float, PRNGKeyArray

from ._system import ContinuousTimeSystem


class ObservationChoice(StrEnum):
    ALL_STATES = "ALL_STATES"
    ALL_POSITIONS = "ALL_POSITIONS"
    LAST_POSITION = "LAST_POSITION"


class ControlChoice(StrEnum):
    ALL_FORCES = "ALL_FORCES"
    LAST_FORCE = "LAST_FORCE"
    NO_CONTROL = "NO_CONTROL"


class MassSpringDamperSystem(ContinuousTimeSystem):
    """A discrete noisy chain with state ``[positions, velocities]``.

    The deterministic matrices are obtained from a continuous model using an
    exact zero-order hold. Noise covariance arguments describe the discrete
    noise added once per simulation step.

    ``observation_choice`` / ``control_choice`` are init-only; the resulting
    matrices are stored on the module.
    """

    number_of_connections: int = eqx.field(static=True, default=1)
    mass: float = eqx.field(static=True, default=1.0)
    spring_constant: float = eqx.field(static=True, default=1.0)
    damping_coefficient: float = eqx.field(static=True, default=1.0)
    initial_state_standard_deviation: float = eqx.field(static=True, default=1.0)

    observation_choice: InitVar[ObservationChoice] = ObservationChoice.LAST_POSITION
    control_choice: InitVar[ControlChoice] = ControlChoice.NO_CONTROL

    time_step: float = eqx.field(static=True, default=0.01)
    state_dim: int = eqx.field(static=True, default=0)
    observation_dim: int = eqx.field(static=True, default=0)
    control_dim: int = eqx.field(static=True, default=0)
    batch_size: int = eqx.field(static=True, default=1)

    continuous_state_matrix: Float[Array, "state_dim state_dim"] = eqx.field(
        default=0.0,
        converter=jnp.asarray,
    )
    continuous_control_matrix: Float[Array, "state_dim control_dim"] = eqx.field(
        default=0.0,
        converter=jnp.asarray,
    )
    observation_matrix: Float[Array, "observation_dim state_dim"] = eqx.field(
        default=0.0,
        converter=jnp.asarray,
    )
    discrete_state_matrix: Float[Array, "state_dim state_dim"] = eqx.field(
        default=0.0,
        converter=jnp.asarray,
    )
    discrete_control_matrix: Float[Array, "state_dim control_dim"] = eqx.field(
        default=0.0,
        converter=jnp.asarray,
    )
    process_noise_covariance: Float[Array, "state_dim state_dim"] = eqx.field(
        default=0.0,
        converter=jnp.asarray,
    )
    observation_noise_covariance: Float[Array, "observation_dim observation_dim"] = eqx.field(
        default=0.0,
        converter=jnp.asarray,
    )

    def __post_init__(
        self,
        observation_choice: ObservationChoice,
        control_choice: ControlChoice,
    ) -> None:
        assert isinstance(observation_choice, ObservationChoice)
        assert isinstance(control_choice, ControlChoice)

        continuous_state_matrix = self._make_state_matrix(
            self.number_of_connections,
            self.mass,
            self.spring_constant,
            self.damping_coefficient,
        )
        continuous_control_matrix = self._make_control_matrix(
            self.number_of_connections,
            self.mass,
            control_choice,
        )
        observation_matrix = self._make_observation_matrix(
            self.number_of_connections,
            observation_choice,
        )
        state_dim, control_dim = continuous_control_matrix.shape
        augmented = jnp.zeros((state_dim + control_dim, state_dim + control_dim))
        augmented = augmented.at[:state_dim, :state_dim].set(continuous_state_matrix)
        augmented = augmented.at[:state_dim, state_dim:].set(continuous_control_matrix)
        exponential = jsp_linalg.expm(augmented * self.time_step)

        self.state_dim = 2 * self.number_of_connections
        self.observation_dim = int(observation_matrix.shape[0])
        self.control_dim = int(control_dim)
        self.continuous_state_matrix = continuous_state_matrix
        self.continuous_control_matrix = continuous_control_matrix
        self.observation_matrix = observation_matrix
        self.discrete_state_matrix = exponential[:state_dim, :state_dim]
        self.discrete_control_matrix = exponential[:state_dim, state_dim:]

        process = jnp.asarray(self.process_noise_covariance)
        if process.ndim == 0:
            process = jnp.broadcast_to(process, (self.state_dim, self.state_dim))
        self.process_noise_covariance = process
        observation = jnp.asarray(self.observation_noise_covariance)
        if observation.ndim == 0:
            observation = jnp.broadcast_to(
                observation,
                (self.observation_dim, self.observation_dim),
            )
        self.observation_noise_covariance = observation

    def __check_init__(self) -> None:
        super().__check_init__()
        assert self.number_of_connections > 0
        assert self.mass > 0
        assert self.spring_constant >= 0
        assert self.damping_coefficient >= 0
        assert self.initial_state_standard_deviation >= 0
        assert self.state_dim == 2 * self.number_of_connections
        assert self.continuous_state_matrix.shape == (self.state_dim, self.state_dim)
        assert self.continuous_control_matrix.shape == (self.state_dim, self.control_dim)
        assert self.observation_matrix.shape == (self.observation_dim, self.state_dim)
        assert self.discrete_state_matrix.shape == (self.state_dim, self.state_dim)
        assert self.discrete_control_matrix.shape == (self.state_dim, self.control_dim)

    def initial_state(self, random_key: PRNGKeyArray | None = None) -> Float[Array, "batch_size state_dim"]:  # noqa: F722
        state = jnp.zeros((self.batch_size, self.state_dim))
        if random_key is None:
            return state
        return state + self.initial_state_standard_deviation * jax.random.normal(random_key, state.shape)

    def observe(
        self,
        time: float,
        state: Float[Array, "... state_dim"],
        random_key: PRNGKeyArray,
    ) -> Float[Array, "... observation_dim"]:
        return state @ self.observation_matrix.T + self._sample_noise(
            random_key,
            self.observation_noise_covariance,
            state.shape[:-1],
        )

    def process(
        self,
        time: float,
        state: Float[Array, "... state_dim"],
        control: Float[Array, "... control_dim"] | None,
        random_key: PRNGKeyArray,
    ) -> tuple[float, Float[Array, "... state_dim"]]:
        next_state = state @ self.discrete_state_matrix.T
        if control is not None:
            next_state += control @ self.discrete_control_matrix.T
        next_state += self._sample_noise(
            random_key,
            self.process_noise_covariance,
            state.shape[:-1],
        )
        return time + self.time_step, next_state

    @staticmethod
    def _sample_noise(
        random_key: PRNGKeyArray,
        covariance: Array,
        batch_shape: tuple[int, ...],
    ) -> Array:
        dimension = covariance.shape[0]
        return jax.lax.cond(
            jnp.all(covariance == 0),
            lambda: jnp.zeros((*batch_shape, dimension)),
            lambda: jax.random.multivariate_normal(
                random_key,
                jnp.zeros(dimension),
                covariance,
                shape=batch_shape,
            ),
        )

    @staticmethod
    def _make_state_matrix(
        count: int,
        mass: float,
        spring: float,
        damping: float,
    ) -> Array:
        stiffness = jnp.zeros((count, count)).at[0, 0].set(spring)
        damping_matrix = jnp.zeros((count, count)).at[0, 0].set(damping)
        for connection in range(1, count):
            indices = jnp.array([connection - 1, connection])
            coupling = jnp.array([[1.0, -1.0], [-1.0, 1.0]])
            stiffness = stiffness.at[jnp.ix_(indices, indices)].add(spring * coupling)
            damping_matrix = damping_matrix.at[jnp.ix_(indices, indices)].add(damping * coupling)
        zeros = jnp.zeros((count, count))
        identity = jnp.eye(count)
        return jnp.block([[zeros, identity], [-stiffness / mass, -damping_matrix / mass]])

    @staticmethod
    def _make_control_matrix(count: int, mass: float, choice: ControlChoice) -> Array:
        match choice:
            case ControlChoice.ALL_FORCES:
                return jnp.concatenate((jnp.zeros((count, count)), jnp.eye(count) / mass), axis=0)
            case ControlChoice.LAST_FORCE:
                matrix = jnp.zeros((2 * count, 1))
                return matrix.at[-1, 0].set(1 / mass)
            case ControlChoice.NO_CONTROL:
                return jnp.zeros((2 * count, 0))

    @staticmethod
    def _make_observation_matrix(count: int, choice: ObservationChoice) -> Array:
        match choice:
            case ObservationChoice.ALL_STATES:
                return jnp.eye(2 * count)
            case ObservationChoice.ALL_POSITIONS:
                return jnp.concatenate((jnp.eye(count), jnp.zeros((count, count))), axis=1)
            case ObservationChoice.LAST_POSITION:
                matrix = jnp.zeros((1, 2 * count))
                return matrix.at[0, count - 1].set(1.0)
