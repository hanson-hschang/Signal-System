import click

from ss.system.dense_state.examples.cart_pole import CartPoleSystem


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
