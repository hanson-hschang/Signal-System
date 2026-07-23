"""Framework for simulating continuous and discrete-time systems."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from copy import copy

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

if TYPE_CHECKING:
    from ss.control._control import Controller, ControllerState


class System(eqx.Module):
    time_step: float = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    observation_dim: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        assert self.time_step >= 0, f"time_step {self.time_step} must be >= 0"
        assert self.state_dim > 0, f"state_dim {self.state_dim} must be > 0"
        assert self.observation_dim > 0, (
            f"observation_dim {self.observation_dim} must be > 0"
        )
        assert self.control_dim >= 0, (
            f"control_dim {self.control_dim} must be >= 0"
        )
        assert self.batch_size > 0, f"batch_size {self.batch_size} must be > 0"

    def duplicate(self, *, batch_size: int) -> Self:
        """Return an immutable copy configured for a new batch size."""
        assert batch_size > 0, f"batch_size {batch_size} must be > 0"
        duplicate = copy(self)
        object.__setattr__(duplicate, "batch_size", batch_size)
        return duplicate

    def init_state(
        self, random_key: PRNGKeyArray | None = None
    ) -> Float[Array, state_dim]:
        return jnp.zeros(self.state_dim)

    def process(
        self,
        time: float,
        state: Float[Array, state_dim],
        control: Float[Array, control_dim],
        random_key: PRNGKeyArray,
    ) -> tuple[float, Float[Array, state_dim]]:
        return (
            time + self.time_step,
            self._state_process(time, state, control)
            + self._process_noise(time, state, random_key),
        )

    def observe(
        self,
        time: float,
        state: Float[Array, state_dim],
        random_key: PRNGKeyArray,
    ) -> Float[Array, observation_dim]:
        return self._observation_process(
            time, state
        ) + self._observation_noise(time, state, random_key)

    def _state_process(
        self, time: float, state: Array, control: Array
    ) -> Float[Array, state_dim]:
        return state

    def _observation_process(
        self, time: float, state: Array
    ) -> Float[Array, observation_dim]:
        return jnp.zeros(self.observation_dim)

    def _process_noise(
        self, time: float, state: Array, random_key: PRNGKeyArray
    ) -> Array:
        return jnp.zeros(self.state_dim)

    def _observation_noise(
        self, time: float, state: Array, random_key: PRNGKeyArray
    ) -> Array:
        return jnp.zeros(self.observation_dim)


class ContinuousTimeSystem(System):
    process_noise_covariance: Float[Array, state_dim state_dim]
    observation_noise_covariance: Float[
        Array, observation_dim observation_dim
    ]

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

    def _process_noise(
        self, time: float, state: Array, random_key: PRNGKeyArray
    ) -> Array:
        cov = self.process_noise_covariance * jnp.sqrt(self.time_step)
        # multivariate_normal requires positive-definite covariance; when
        # covariance is identically zero (no noise requested) skip sampling
        # entirely rather than producing NaN.
        return jax.lax.cond(
            jnp.all(cov == 0),
            lambda: jnp.zeros(self.state_dim),
            lambda: jax.random.multivariate_normal(
                random_key, jnp.zeros(self.state_dim), cov
            ),
        )

    def _observation_noise(
        self, time: float, state: Array, random_key: PRNGKeyArray
    ) -> Array:
        cov = self.observation_noise_covariance * jnp.sqrt(self.time_step)
        return jax.lax.cond(
            jnp.all(cov == 0),
            lambda: jnp.zeros(self.observation_dim),
            lambda: jax.random.multivariate_normal(
                random_key, jnp.zeros(self.observation_dim), cov
            ),
        )


class DiscreteTimeSystem(ContinuousTimeSystem):
    def __check_init__(self) -> None:
        super().__check_init__()
        assert self.time_step == 1, (
            "DiscreteTimeSystem requires time_step == 1"
        )


SystemT = TypeVar("SystemT", bound=System)


def simulate(
    system: SystemT,
    initial_time: Float,
    initial_state: Array,
    keys: PRNGKeyArray,
    controller: Controller | None = None,
) -> tuple[Array, Array, Array, Array | None]:
    """Simulate batches with a time scan containing vmapped system steps."""
    if controller is not None:
        assert system.batch_size == controller.batch_size  # TODO: message
        controller_state = controller.init_state()  # TODO: key?
    else:
        controller_state = None

    @jax.jit
    def body(
        carry: tuple[
            Float,  # time
            Array,  # system state
            ControllerState | None,  # controller state
        ],
        random_key: PRNGKeyArray,
    ) -> tuple[
        tuple[Float, Array, ControllerState | None],
        tuple[Float, Array, Array, Array | None],
    ]:
        previous_time, previous_state, controller_state = carry

        # Key splits
        step_keys = jax.random.split(random_key, 2 * system.batch_size + 1)
        observe_keys = step_keys[: system.batch_size]
        process_keys = step_keys[system.batch_size : 2 * system.batch_size]
        controller_key = step_keys[-1]

        observation = system.observe(
            previous_time, previous_state, observe_keys
        )

        if controller is None:
            control = None
            next_controller_state = None
            next_time, state = system.process(
                previous_time, previous_state, process_keys
            )
        else:
            control, next_controller_state, _ = controller(
                controller_state,
                previous_time,
                observation,
                controller_key,
            )
            next_time, state = system.process(
                previous_time, previous_state, process_keys, control=control
            )

        return (
            next_time,
            state,
            next_controller_state,
        ), (next_time, state, observation, control)

    _, (times, states, observations, controls) = jax.lax.scan(
        body,
        (initial_time, initial_state, controller_state),
        keys,
    )

    return times, states, observations, controls
