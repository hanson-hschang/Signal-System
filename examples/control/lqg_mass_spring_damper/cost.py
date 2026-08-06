# ruff: noqa: F722

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class QuadraticCost(eqx.Module):
    state_weight: Float[Array, "state_dim state_dim"]
    control_weight: Float[Array, "control_dim control_dim"]

    def running_cost(
        self,
        state: Float[Array, "*batch state_dim"],
        control: Float[Array, "*batch control_dim"],
    ) -> Float[Array, "*batch"]:
        state_cost = jnp.einsum("...i,ij,...j->...", state, self.state_weight, state)
        control_cost = jnp.einsum("...i,ij,...j->...", control, self.control_weight, control)
        return 0.5 * (state_cost + control_cost)
