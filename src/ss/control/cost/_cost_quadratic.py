"""Quadratic state/control costs."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class QuadraticCost(eqx.Module):
    """Batched quadratic cost ``0.5 (x' Q x + u' R u)`` about the origin.

    ``terminal_cost`` reuses the state quadratic form scaled by
    ``terminal_scale``. Controllers that only need running costs (e.g. LQG
    gain design) can ignore the terminal method.
    """

    state_weight: Float[Array, "state_dim state_dim"] = eqx.field(converter=jnp.asarray)
    control_weight: Float[Array, "control_dim control_dim"] = eqx.field(converter=jnp.asarray)
    terminal_scale: float = eqx.field(static=True, default=1.0)

    def __check_init__(self) -> None:
        assert self.state_weight.ndim == 2
        assert self.control_weight.ndim == 2
        assert self.state_weight.shape[0] == self.state_weight.shape[1]
        assert self.control_weight.shape[0] == self.control_weight.shape[1]
        assert self.terminal_scale >= 0
        assert jnp.allclose(self.state_weight, self.state_weight.T), "state_weight must be symmetric"
        assert jnp.allclose(self.control_weight, self.control_weight.T), "control_weight must be symmetric"
        assert jnp.all(jnp.linalg.eigvalsh(self.state_weight) >= 0), "state_weight must be positive semidefinite"
        assert jnp.all(jnp.linalg.eigvalsh(self.control_weight) >= 0), "control_weight must be positive semidefinite"

    def state_cost(self, state: Float[Array, "*batch state_dim"]) -> Float[Array, "*batch"]:
        return 0.5 * jnp.einsum("...i,ij,...j->...", state, self.state_weight, state)

    def running_cost(
        self,
        state: Float[Array, "*batch state_dim"],
        control: Float[Array, "*batch control_dim"],
    ) -> Float[Array, "*batch"]:
        control_cost = 0.5 * jnp.einsum(
            "...i,ij,...j->...",
            control,
            self.control_weight,
            control,
        )
        return self.state_cost(state) + control_cost

    def terminal_cost(self, state: Float[Array, "*batch state_dim"]) -> Float[Array, "*batch"]:
        return self.terminal_scale * self.state_cost(state)
