"""Framework for simulating continuous and discrete-time systems."""

from __future__ import annotations

from abc import abstractmethod
from copy import copy
from typing import TYPE_CHECKING, TypeVar, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, Shaped

if TYPE_CHECKING:
    from ss.control._control import Controller, ControllerState, Diagnostics


class System(eqx.Module):
    time_step: float = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    observation_dim: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        assert self.time_step >= 0, f"time_step {self.time_step} must be >= 0"
        assert self.state_dim > 0, f"state_dim {self.state_dim} must be > 0"
        assert self.observation_dim > 0, f"observation_dim {self.observation_dim} must be > 0"
        assert self.control_dim >= 0, f"control_dim {self.control_dim} must be >= 0"
        assert self.batch_size > 0, f"batch_size {self.batch_size} must be > 0"

    def duplicate(self, *, batch_size: int | None = None) -> Self:
        """Return an immutable copy configured for a new batch size."""
        # TODO: This is temporary duplication for different batch size.
        # Mostly used for testing and demonstration, but may need to change the
        # syntax in the future.
        if batch_size is None:
            batch_size = self.batch_size
        assert batch_size > 0, f"batch_size {batch_size} must be > 0"
        duplicate = copy(self)
        object.__setattr__(duplicate, "batch_size", batch_size)
        return duplicate

    @abstractmethod
    def initial_state(self, random_key: PRNGKeyArray | None = None) -> Float[Array, "batch_size state_dim"]:
        pass

    @abstractmethod
    def process(
        self,
        time: float,
        state: Float[Array, "batch_size state_dim"],
        control: Float[Array, "batch_size control_dim"] | None,
        random_key: PRNGKeyArray,
    ) -> tuple[float, Float[Array, "batch_size state_dim"]]:
        pass

    @abstractmethod
    def observe(
        self,
        time: float,
        state: Float[Array, "batch_size state_dim"],
        random_key: PRNGKeyArray,
    ) -> Float[Array, "batch_size observation_dim"]:
        pass


class ContinuousTimeSystem(System):
    process_noise_covariance: Float[Array, "state_dim state_dim"]
    observation_noise_covariance: Float[Array, "observation_dim observation_dim"]

    def __check_init__(self) -> None:
        super().__check_init__()
        s = (self.state_dim, self.state_dim)
        o = (self.observation_dim, self.observation_dim)
        assert self.process_noise_covariance.shape == s, (
            f"process_noise_covariance must have shape {s}, got {self.process_noise_covariance.shape}"
        )
        assert self.observation_noise_covariance.shape == o, (
            f"observation_noise_covariance must have shape {o}, got {self.observation_noise_covariance.shape}"
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


SystemT = TypeVar("SystemT", bound=System)


class SimulateCarry(eqx.Module):
    time: float
    state: Shaped[Array, "batch_size ..."]
    controller_state: ControllerState | None


type SimulateStep = tuple[
    float,  # time
    Shaped[Array, "batch_size ..."],  # state
    Shaped[Array, "batch_size observation_dim"],  # observation
    Float[Array, "batch_size control_dim"] | None,  # control
    Diagnostics | None,  # controller diagnostics
]


class SimulationResult(eqx.Module):
    """Time-aligned system trajectory and per-step controller outputs."""

    times: Float[Array, "state_time"]
    states: Shaped[Array, "state_time batch_size ..."]
    observations: Shaped[Array, "step_time batch_size observation_dim"]
    controls: Float[Array, "step_time batch_size control_dim"] | None
    controller_diagnostics: Diagnostics | None


def simulate(
    system: SystemT,
    initial_time: float,
    number_of_steps: int,
    initial_state: Shaped[Array, "batch_size ..."],
    random_key: PRNGKeyArray | None = None,
    controller: Controller | None = None,
) -> SimulationResult:
    """Simulate batches with a time scan containing system steps.

    Layout matches ``filtering`` / ``lax.scan``: time axis first, then batch.
    Dim names come from ``SystemT`` (``batch_size``, ``observation_dim``,
    ``control_dim``); state trailing dims vary by concrete system.

    Args:
        system: System to simulate.
        initial_time: Time before the first step.
        number_of_steps: Number of observe/process steps to run.
        initial_state: Initial system state, shape ``(batch_size, ...)``.
        random_key: PRNG key. If ``None``, a fixed default key is used.
        controller: Optional controller applied each step.

    Returns:
        A result whose ``times`` and ``states`` include the initial and final
        samples. Observations, controls, and diagnostics describe each
        transition and therefore contain ``number_of_steps`` samples.
    """
    assert number_of_steps > 0, f"number_of_steps {number_of_steps} must be > 0"
    if random_key is None:
        random_key = jax.random.PRNGKey(43)

    if controller is not None:
        assert system.batch_size == controller.batch_size, (
            f"system.batch_size {system.batch_size} must match controller.batch_size {controller.batch_size}"
        )
        assert system.control_dim == controller.control_dim, (
            f"system.control_dim {system.control_dim} must match controller.control_dim {controller.control_dim}"
        )
        random_key, controller_key = jax.random.split(random_key)
        controller_state = controller.initial_state(controller_key)
    else:
        controller_state = None

    keys = jax.random.split(random_key, number_of_steps)

    @jax.jit
    def step(
        carry: SimulateCarry,
        random_key: PRNGKeyArray,
    ) -> tuple[SimulateCarry, SimulateStep]:
        observe_key, process_key, controller_key = jax.random.split(random_key, 3)

        observation = system.observe(carry.time, carry.state, observe_key)

        if controller is None:
            control = None
            next_controller_state = None
            diagnostics = None
            next_time, state = system.process(carry.time, carry.state, control, process_key)
        else:
            control, next_controller_state, diagnostics = controller(
                carry.controller_state,
                carry.time,
                observation,
                controller_key,
            )
            next_time, state = system.process(carry.time, carry.state, control, process_key)

        return SimulateCarry(next_time, state, next_controller_state), (
            carry.time,
            carry.state,
            observation,
            control,
            diagnostics,
        )

    final_carry, outputs = jax.lax.scan(
        step,
        SimulateCarry(initial_time, initial_state, controller_state),
        keys,
    )
    times, states, observations, controls, diagnostics = outputs
    times = jnp.concatenate((times, jnp.asarray(final_carry.time)[None]))
    states = jnp.concatenate((states, final_carry.state[None]), axis=0)

    return SimulationResult(
        times,
        states,
        observations,
        controls,
        diagnostics,
    )
