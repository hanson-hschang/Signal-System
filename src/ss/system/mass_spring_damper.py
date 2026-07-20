# ruff: noqa: F722, F821

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
    """A wall-connected chain with state ``[positions, velocities]``."""

    number_of_connections: int = eqx.field(static=True)
    mass: float = eqx.field(static=True)
    spring_constant: float = eqx.field(static=True)
    damping_coefficient: float = eqx.field(static=True)
    initial_state_standard_deviation: float = eqx.field(static=True)

    continuous_state_matrix: Float[Array, "state_dim state_dim"]
    continuous_control_matrix: Float[Array, "state_dim control_dim"]
    observation_matrix: Float[Array, "observation_dim state_dim"]
    discrete_state_matrix: Float[Array, "state_dim state_dim"]
    discrete_control_matrix: Float[Array, "state_dim control_dim"]

    def __init__(
        self,
        number_of_connections: int = 1,
        mass: float = 1.0,
        spring_constant: float = 1.0,
        damping_coefficient: float = 1.0,
        time_step: float = 0.01,
        observation_choice: ObservationChoice = (
            ObservationChoice.LAST_POSITION
        ),
        control_choice: ControlChoice = ControlChoice.NO_CONTROL,
        process_noise_covariance: Array | None = None,
        observation_noise_covariance: Array | None = None,
        initial_state_standard_deviation: float = 1.0,
        batch_size: int = 1,
    ) -> None:
        assert number_of_connections > 0
        assert mass > 0
        assert spring_constant >= 0
        assert damping_coefficient >= 0
        assert time_step > 0
        assert initial_state_standard_deviation >= 0
        assert isinstance(observation_choice, ObservationChoice)
        assert isinstance(control_choice, ControlChoice)

        state_dim = 2 * number_of_connections
        state_matrix = self._make_state_matrix(
            number_of_connections,
            mass,
            spring_constant,
            damping_coefficient,
        )
        control_matrix = self._make_control_matrix(
            number_of_connections, mass, control_choice
        )
        observation_matrix = self._make_observation_matrix(
            number_of_connections, observation_choice
        )
        discrete_state_matrix, discrete_control_matrix = self._discretize(
            state_matrix, control_matrix, time_step
        )

        self.time_step = time_step
        self.state_dim = state_dim
        self.observation_dim = observation_matrix.shape[0]
        self.control_dim = control_matrix.shape[1]
        self.batch_size = batch_size
        self.number_of_connections = number_of_connections
        self.mass = mass
        self.spring_constant = spring_constant
        self.damping_coefficient = damping_coefficient
        self.initial_state_standard_deviation = (
            initial_state_standard_deviation
        )
        self.continuous_state_matrix = state_matrix
        self.continuous_control_matrix = control_matrix
        self.observation_matrix = observation_matrix
        self.discrete_state_matrix = discrete_state_matrix
        self.discrete_control_matrix = discrete_control_matrix
        self.process_noise_covariance = self._covariance_or_zeros(
            process_noise_covariance, state_dim
        )
        self.observation_noise_covariance = self._covariance_or_zeros(
            observation_noise_covariance, self.observation_dim
        )

    def __check_init__(self) -> None:
        super().__check_init__()
        assert self.number_of_connections > 0
        assert self.mass > 0
        assert self.spring_constant >= 0
        assert self.damping_coefficient >= 0
        assert self.initial_state_standard_deviation >= 0

    def init_state(
        self, random_key: PRNGKeyArray | None = None
    ) -> Float[Array, "batch_size state_dim"]:  # noqa: F722
        state = jnp.zeros((self.batch_size, self.state_dim))
        if random_key is None:
            return state
        return (
            state
            + self.initial_state_standard_deviation
            * jax.random.normal(random_key, state.shape)
        )

    def observe(
        self,
        time: float,
        state: Float[Array, "state_dim"],
        random_key: PRNGKeyArray,
    ) -> Float[Array, "observation_dim"]:
        return self.observation_matrix @ state + self._observation_noise(
            time, state, random_key
        )

    def process(
        self,
        time: float,
        state: Float[Array, "state_dim"],
        random_key: PRNGKeyArray,
        *,
        control: Float[Array, "control_dim"] | None = None,
    ) -> tuple[float, Float[Array, "state_dim"]]:
        next_state = self.discrete_state_matrix @ state
        if control is not None:
            next_state += self.discrete_control_matrix @ control
        next_state += self._process_noise(time, state, random_key)
        return time + self.time_step, next_state

    @staticmethod
    def _covariance_or_zeros(
        covariance: Array | None, dimension: int
    ) -> Array:
        if covariance is None:
            return jnp.zeros((dimension, dimension))
        return jnp.asarray(covariance)

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
            stiffness = stiffness.at[jnp.ix_(indices, indices)].add(
                spring * coupling
            )
            damping_matrix = damping_matrix.at[jnp.ix_(indices, indices)].add(
                damping * coupling
            )
        zeros = jnp.zeros((count, count))
        identity = jnp.eye(count)
        return jnp.block(
            [[zeros, identity], [-stiffness / mass, -damping_matrix / mass]]
        )

    @staticmethod
    def _make_control_matrix(
        count: int, mass: float, choice: ControlChoice
    ) -> Array:
        if choice == ControlChoice.ALL_FORCES:
            return jnp.concatenate(
                (jnp.zeros((count, count)), jnp.eye(count) / mass), axis=0
            )
        if choice == ControlChoice.LAST_FORCE:
            matrix = jnp.zeros((2 * count, 1))
            return matrix.at[-1, 0].set(1 / mass)
        return jnp.zeros((2 * count, 0))

    @staticmethod
    def _make_observation_matrix(
        count: int, choice: ObservationChoice
    ) -> Array:
        if choice == ObservationChoice.ALL_STATES:
            return jnp.eye(2 * count)
        if choice == ObservationChoice.ALL_POSITIONS:
            return jnp.concatenate(
                (jnp.eye(count), jnp.zeros((count, count))), axis=1
            )
        matrix = jnp.zeros((1, 2 * count))
        return matrix.at[0, count - 1].set(1.0)

    @staticmethod
    def _discretize(
        state_matrix: Array, control_matrix: Array, time_step: float
    ) -> tuple[Array, Array]:
        state_dim, control_dim = control_matrix.shape
        augmented = jnp.zeros(
            (state_dim + control_dim, state_dim + control_dim)
        )
        augmented = augmented.at[:state_dim, :state_dim].set(state_matrix)
        augmented = augmented.at[:state_dim, state_dim:].set(control_matrix)
        exponential = jsp_linalg.expm(augmented * time_step)
        return (
            exponential[:state_dim, :state_dim],
            exponential[:state_dim, state_dim:],
        )
