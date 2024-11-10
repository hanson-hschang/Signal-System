import click
import numpy as np
from numba import njit
from numpy.typing import NDArray

from system.dense_state.nonlinear import ContinuousTimeNonlinearSystem


class CartPoleSystem(ContinuousTimeNonlinearSystem):
    """
    Cart-pole system dynamics.

    The cart-pole system is a classic control problem where a pole is attached to a cart moving along a frictionless track.

    The number of states is 4, with the state vector including position and velocity of the cart, and angle and angular velocity of the pole.
    The pole is at its upright position when the angle is 0, and the positive direction is counter-clockwise.
    The number of observations is 4, with the observation vector being the same as the state vector.
    The number of controls is 1, with the control vector being the force applied to the cart.

    The system is described by the equations from: https://courses.ece.ucsb.edu/ECE594/594D_W10Byl/hw/cartpole_eom.pdf
    """

    def __init__(
        self,
        cart_mass: float = 1.0,
        pole_mass: float = 0.01,
        pole_length: float = 2.0,
        gravity: float = 9.81,
        time_step: float = 0.01,
        number_of_systems: int = 1,
    ) -> None:
        self._cart_mass = cart_mass
        self._pole_mass = pole_mass
        self._pole_length = pole_length
        self._gravity = gravity

        @njit(cache=True)  # type: ignore
        def process_function(
            state: NDArray[np.float64],
            control: NDArray[np.float64],
            cart_mass: float = self._cart_mass,
            pole_mass: float = self._pole_mass,
            pole_length: float = self._pole_length,
            gravity: float = self._gravity,
        ) -> NDArray[np.float64]:
            cart_position = state[:, 0]
            cart_velocity = state[:, 1]
            pole_angle = state[:, 2]
            pole_angular_velocity = state[:, 3]
            force = control[:, 0]
            process = np.zeros_like(state)

            total_mass = cart_mass + pole_mass
            total_mass_adjusted = cart_mass + (
                pole_mass * np.sin(pole_angle) ** 2
            )
            pole_mass_length = pole_mass * pole_length

            common_numerator = force - (
                pole_mass_length
                * np.sin(pole_angle)
                * (pole_angular_velocity**2)
            )
            process[:, 3] = (
                common_numerator * np.cos(pole_angle)
                + total_mass * gravity * np.sin(pole_angle)
            ) / (total_mass_adjusted * pole_length)
            process[:, 2] = pole_angular_velocity
            process[:, 1] = (
                common_numerator
                + pole_mass * gravity * np.sin(pole_angle) * np.cos(pole_angle)
            ) / total_mass_adjusted
            process[:, 0] = cart_velocity

            return process

        @njit(cache=True)  # type: ignore
        def observation_function(
            state: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            return state[:, :]

        super().__init__(
            time_step=time_step,
            state_dim=4,
            observation_dim=4,
            control_dim=1,
            process_function=process_function,
            observation_function=observation_function,
            number_of_systems=number_of_systems,
        )


@click.command()
@click.option(
    "--cart-mass",
    type=click.FloatRange(min=0),
    default=1.0,
    help="Set the mass of the cart (positive value).",
)
@click.option(
    "--pole-mass",
    type=click.FloatRange(min=0),
    default=0.01,
    help="Set the mass of the pole (positive value).",
)
@click.option(
    "--pole-length",
    type=click.FloatRange(min=0),
    default=2.0,
    help="Set the length of the pole (positive value).",
)
@click.option(
    "--gravity",
    type=float,
    default=9.81,
    help="Set the value of gravity (positive value).",
)
@click.option(
    "--time-step",
    type=click.FloatRange(min=0),
    default=0.01,
    help="Set the time step (positive value).",
)
@click.option(
    "--number-of-systems",
    type=click.IntRange(min=1),
    default=1,
    help="Set the number of systems (positive integers).",
)
def main(
    cart_mass: float,
    pole_mass: float,
    pole_length: float,
    gravity: float,
    time_step: float,
    number_of_systems: int,
) -> None:
    cart_pole_system = CartPoleSystem(
        cart_mass=cart_mass,
        pole_mass=pole_mass,
        pole_length=pole_length,
        gravity=gravity,
        time_step=time_step,
        number_of_systems=number_of_systems,
    )
    if number_of_systems > 1:
        cart_pole_system.control = [[1]] * number_of_systems
    else:
        cart_pole_system.control = [1]
    time = cart_pole_system.process(0)
    print(cart_pole_system.state)
    observation = cart_pole_system.observe()
    print(observation)


if __name__ == "__main__":
    main()
