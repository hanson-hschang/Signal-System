
from __future__ import annotations

from functools import partial
from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class Filter(eqx.Module):

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

    def init_state(self) -> Float[Array, "state_dim"]:
        return jnp.zeros(self.state_dim)

    def update(
        self,
        time: Float,
        prior: Float[Array, "state_dim"],
        observation: Float[Array, "observation_dim"],
        # control: Float[Array, "control_dim"],
    ) -> tuple[Float, Float[Array, "state_dim"]]:
        posterior = prior
        return (
            time + self.time_step,
            posterior
        )

FilterT = TypeVar("FilterT", bound=Filter)


def filtering_step(
    filter: FilterT,
    carry: tuple[Float, Array],
    observation: Array,
) -> tuple[tuple[Float, Array], tuple[Float, Array]]:
    previous_time, previous_belief = carry

    time, belief = filter.update(previous_time, previous_belief, observation)
    return (time, belief), (time, belief)


def filtering(
    filter: FilterT,
    initial_time: Float,
    initial_belief: Array,
    observations: Array,
) -> tuple[Array, Array]:
    """Run a filtering algorithm on a sequence of observations.

    Args:
        filter: The filter to use for the filtering process.
        initial_time: The initial time before processing any observations.
        initial_belief: The initial belief state before processing any observations.
        observations: A sequence of observations to process.

    Returns:
        A tuple containing:
            - An array of times corresponding to each observation processed.
            - An array of beliefs corresponding to each observation processed.
    """

    body = partial(filtering_step, filter)

    _, (times, beliefs) = jax.lax.scan(
        body,
        (initial_time, initial_belief),
        observations
    )

    if beliefs.ndim == 1:
        beliefs = beliefs[:, jnp.newaxis]

    return times, beliefs

def batch_filtering(
    filter: FilterT,
    initial_time: Float,
    initial_beliefs: Array,  # (batch, state_dim,)
    observations: Array,  # (batch, time_horizon, observation_dim,)
) -> tuple[Array, Array]:
    """Run a filtering algorithm on a batch of sequences of observations.

    Args:
        filter: The filter to use for the filtering process.
        initial_time: The initial time before processing any observations.
        initial_beliefs: An array of initial belief states for each sequence in the batch.
        observations: A batch of sequences of observations to process.

    Returns:
        A tuple containing:
            - An array of times corresponding to each observation processed for each sequence in the batch.
            - An array of beliefs corresponding to each observation processed for each sequence in the batch.
    """

    _batch_filtering = jax.vmap(
        filtering,
        in_axes=(None, None, 0, 0),  # filter is static, initial_time is static, initial_beliefs and observations are batched
    )

    return _batch_filtering(
        filter, initial_time, initial_beliefs, observations
    )
