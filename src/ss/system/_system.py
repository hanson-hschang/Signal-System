"""
A dynamical system framework for simulating continuous and discrete time systems.
"""
from __future__ import annotations

from typing import Callable, TypeVar
from functools import partial
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class System(eqx.Module):

    time_step: float = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    observation_dim: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        assert self.time_step >= 0, f"time_step {self.time_step} must be >= 0"
        assert self.state_dim > 0, f"state_dim {self.state_dim} must be > 0"
        assert self.observation_dim > 0, (
            f"observation_dim {self.observation_dim} must be > 0"
        )
        assert self.control_dim >= 0, (
            f"control_dim {self.control_dim} must be >= 0"
        )

    def init_state(
        self, random_key: PRNGKeyArray | None = None
    ) -> Float[Array, state_dim]:
        return jnp.zeros(self.state_dim)

    def process(
        self,
        time: float,
        state: Float[Array, "state_dim"],
        control: Float[Array, "control_dim"],
        random_key: PRNGKeyArray,
    ) -> tuple[float, Float[Array, "state_dim"]]:
        return (
            time + self.time_step,
            self._state_process(time, state, control) + self._process_noise(time, state, random_key)
        )

    def observe(
        self,
        time: float,
        state: Float[Array, "state_dim"],
        random_key: PRNGKeyArray,
    ) -> Float[Array, "observation_dim"]:
        return self._observation_process(time, state) + self._observation_noise(
            time, state, random_key
        )

    def _state_process(
        self, time: float, state: Array, control: Array
    ) -> Float[Array, "state_dim"]:
        return state

    def _observation_process(self, time: float, state: Array) -> Float[Array, "observation_dim"]:
        return jnp.zeros(self.observation_dim)

    def _process_noise(self, time: float, state: Array, random_key: PRNGKeyArray) -> Array:
        return jnp.zeros(self.state_dim)

    def _observation_noise(self, time: float, state: Array, random_key: PRNGKeyArray) -> Array:
        return jnp.zeros(self.observation_dim)


class ContinuousTimeSystem(System):

    process_noise_covariance: Float[Array, "state_dim state_dim"]
    observation_noise_covariance: Float[Array, "observation_dim observation_dim"]

    def __check_init__(self) -> None:
        super().__check_init__()
        s = (self.state_dim, self.state_dim)
        o = (self.observation_dim, self.observation_dim)
        assert self.process_noise_covariance.shape == s, (
            f"process_noise_covariance must have shape {s}, got "
            f"{self.process_noise_covariance.shape}"
        )
        assert self.observation_noise_covariance.shape == o, (
            f"observation_noise_covariance must have shape {o}, got "
            f"{self.observation_noise_covariance.shape}"
        )

    def _process_noise(self, time: float, state: Array, random_key: PRNGKeyArray) -> Array:
        cov = self.process_noise_covariance * jnp.sqrt(self.time_step)
        # multivariate_normal requires positive-definite covariance; when
        # covariance is identically zero (no noise requested) skip sampling
        # entirely rather than producing NaN.
        return jax.lax.cond(
            jnp.all(cov == 0),
            lambda: jnp.zeros(self.state_dim),
            lambda: jax.random.multivariate_normal(random_key, jnp.zeros(self.state_dim), cov),
        )

    def _observation_noise(self, time: float, state: Array, random_key: PRNGKeyArray) -> Array:
        cov = self.observation_noise_covariance * jnp.sqrt(self.time_step)
        return jax.lax.cond(
            jnp.all(cov == 0),
            lambda: jnp.zeros(self.observation_dim),
            lambda: jax.random.multivariate_normal(random_key, jnp.zeros(self.observation_dim), cov),
        )


class DiscreteTimeSystem(ContinuousTimeSystem):
    def __check_init__(self) -> None:
        super().__check_init__()
        assert self.time_step == 1, "DiscreteTimeSystem requires time_step == 1"


SystemT = TypeVar("SystemT", bound=System)


def simulate(
    system: SystemT,
    initial_time: Float,
    initial_state: Array,
    keys: PRNGKeyArray,
    control_policy: Callable[[Float, Array], Array] | None = None,
) -> tuple[Array, Array, Array, Array | None]:
    """
    Simulate from an initial state using a sequence of random keys.

    The control policy receives the current time and observation and returns
    the control input.
    Pass ``control_policy=None`` to use a system's uncontrolled process path.
    """

    def body(
        carry: tuple[Float, Array],
        random_key: PRNGKeyArray,
    ) -> tuple[tuple[Float, Array], tuple[Float, Array, Array, Array | None]]:
        previous_time, previous_state = carry
        process_key, observe_key = jax.random.split(random_key, 2)

        observation = system.observe(
            previous_time, previous_state, observe_key
        )
        if control_policy is None:
            control = None
            time, state = system.process(
                time=previous_time,
                state=previous_state,
                random_key=process_key,
            )
        else:
            control = control_policy(previous_time, observation)
            time, state = system.process(
                time=previous_time,
                state=previous_state,
                control=control,
                random_key=process_key,
            )
        return (time, state), (time, state, observation, control)

    _, (times, states, observations, controls) = jax.lax.scan(
        body, (initial_time, initial_state), keys
    )

    if states.ndim == 1:
        states = states[:, jnp.newaxis]

    if observations.ndim == 1:
        observations = observations[:, jnp.newaxis]

    return times, states, observations, controls


def batch_simulate(
    system: SystemT,
    initial_time: Float,
    initial_states: Array,  # (batch, state_dim,)
    keys: PRNGKeyArray,  # (batch, num_steps,)
    control_policy: Callable[[Float, Array], Array] = None,
) -> tuple[Array, Array, Array, Array]:
    """
    Simulate a batch of systems from the supplied initial states.

    The control policy receives the current time and observation and returns
    the control input.
    """
    _batch_simulate = jax.vmap(simulate, in_axes=(None, None, 0, 0, None))
    return _batch_simulate(
        system, initial_time, initial_states, keys, control_policy
    )
