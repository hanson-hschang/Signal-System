import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

from ss.system import System


class CartPoleSystem(System):
    """
    Cart-pole system dynamics.

    The cart-pole system is a classic control problem where a pole is attached
      to a cart moving along a frictionless track.

    The number of states is 4, with the state vector including position and
    velocity of the cart, and angle and angular velocity of the pole.
    The pole is at its upright position when the angle is 0, and the positive
    direction is counter-clockwise. The number of observations is 4, with the
    observation vector being the same as the state vector. The number of
    controls is 1, with the control vector being the force applied to the cart.

    The system is described by the equations from:
    https://courses.ece.ucsb.edu/ECE594/594D_W10Byl/hw/cartpole_eom.pdf
    """

    time_step: float = eqx.field(static=True, default=0.001)
    state_dim: int = eqx.field(static=True, default=4)
    observation_dim: int = eqx.field(static=True, default=4)
    control_dim: int = eqx.field(static=True, default=1)

    cart_mass: float = 1.0
    pole_mass: float = 0.01
    pole_length: float = 2.0
    gravity: float = 9.81
    batch_size: int = eqx.field(static=True, default=1)

    def __check_init__(self) -> None:
        super().__check_init__()
        assert self.time_step > 0, "time_step must be > 0"
        assert self.cart_mass > 0, "cart_mass must be > 0"
        assert self.pole_mass > 0, "pole_mass must be > 0"
        assert self.pole_length > 0, "pole_length must be > 0"
        assert self.gravity >= 0, "gravity must be >= 0"
        assert self.batch_size > 0, "batch_size must be > 0"

    def init_state(
        self, random_key: PRNGKeyArray | None = None
    ) -> Float[Array, "batch_size state_dim"]:  # noqa: F722
        """
        Initialize near the unstable upright equilibrium.

        If random_key is provided, add stochasticity
        """
        state = jnp.broadcast_to(
            jnp.array([0.0, 0.0, 0.0, 0.0]),
            (self.batch_size, self.state_dim),
        )

        # Stochastic init
        if random_key is None:
            return state

        # FIXME: arbitrary for now.
        standard_deviation = jnp.array([0.05, 0.02, 0.05, 0.02])
        return state + standard_deviation * jax.random.normal(
            random_key, shape=(self.batch_size, self.state_dim)
        )

    def observe(
        self,
        time: float,
        state: Float[Array, "batch_size state_dim"],  # noqa: F722
        random_key: PRNGKeyArray,
    ) -> Float[Array, "batch_size observation_dim"]:  # noqa: F722
        return state

    def process(
        self,
        time: Array,
        state: Array,
        random_key: PRNGKeyArray,
        *,
        control: Array | None = None,
    ) -> tuple[Array, Array]:
        # NOTE: Easy RK4 step impl. Consider diffrax.Tsit5 :)
        half_step = 0.5 * self.time_step
        k1 = self.dynamics(time, state, control)
        k2 = self.dynamics(time + half_step, state + half_step * k1, control)
        k3 = self.dynamics(time + half_step, state + half_step * k2, control)
        k4 = self.dynamics(
            time + self.time_step,
            state + self.time_step * k3,
            control,
        )
        next_state = state + self.time_step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return time + self.time_step, next_state

    def dynamics(
        self,
        time: float,
        state: Array,
        control: Array | None = None,
    ) -> Float[Array, "batch_size state_dim"]:  # noqa: F722
        """
        Cart-pole physics ODEs
        """
        cart_velocity = state[:, 1]
        pole_angle = state[:, 2]
        pole_angular_velocity = state[:, 3]

        total_mass = self.cart_mass + self.pole_mass
        total_mass_adjusted = self.cart_mass + (
            self.pole_mass * jnp.sin(pole_angle) ** 2
        )
        pole_mass_length = self.pole_mass * self.pole_length

        common_numerator = (
            pole_mass_length * jnp.sin(pole_angle) * pole_angular_velocity**2
        )
        pole_angular_acceleration_numerator = (
            total_mass * self.gravity * jnp.sin(pole_angle)
            - pole_mass_length
            * pole_angular_velocity** 2
            * jnp.sin(pole_angle)
            * jnp.cos(pole_angle)
        )
        # This is intended branching: JAX expression saves for no-control
        if control is not None:
            force = control[:, 0]
            common_numerator = common_numerator + force
            pole_angular_acceleration_numerator = (
                pole_angular_acceleration_numerator
                - force * jnp.cos(pole_angle)
            )

        pole_angular_acceleration = pole_angular_acceleration_numerator / (
            total_mass_adjusted * self.pole_length
        )
        cart_acceleration = (
            common_numerator
            - self.pole_mass
            * self.gravity
            * jnp.sin(pole_angle)
            * jnp.cos(pole_angle)
        ) / total_mass_adjusted

        return jnp.stack(
            (
                cart_velocity,
                cart_acceleration,
                pole_angular_velocity,
                pole_angular_acceleration,
            ),
            axis=-1,
        )
