import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array


class CostWeights(eqx.Module):
    cart_position: float = 2.0
    cart_velocity: float = 0.2
    pole_angle: float = 12.0
    pole_angular_velocity: float = 0.5
    control: float = 0.002
    terminal_scale: float = 10.0

    def state_cost(self, state: Array) -> Array:
        """Quadratic state cost about the upright origin."""
        angle_error = jnp.arctan2(jnp.sin(state[..., 2]), jnp.cos(state[..., 2]))
        return (
            self.cart_position * state[..., 0] ** 2
            + self.cart_velocity * state[..., 1] ** 2
            + self.pole_angle * angle_error**2
            + self.pole_angular_velocity * state[..., 3] ** 2
        )

    def running_cost(self, state: Array, control: Array) -> Array:
        return self.state_cost(state) + self.control * control[..., 0] ** 2

    def terminal_cost(self, state: Array) -> Array:
        return self.terminal_scale * self.state_cost(state)
