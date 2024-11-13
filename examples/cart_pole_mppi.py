import click
import numpy as np
from cart_pole import CartPoleSystem, plot_cart_pole_states

from ss.control.cost.quadratic import QuadraticCost
from ss.control.mppi import ModelPredictivePathIntegralController


@click.command()
@click.option(
    "--simulation-time",
    type=click.FloatRange(min=0),
    default=4.0,
    help="Set the simulation time (positive value).",
)
@click.option(
    "--time-step",
    type=click.FloatRange(min=0),
    default=0.02,
    help="Set the time step (positive value).",
)
def main(simulation_time: float, time_step: float) -> None:
    simulation_time_steps = int(simulation_time / time_step)
    system = CartPoleSystem(
        cart_mass=1.0,
        pole_mass=0.01,
        pole_length=2.0,
        gravity=9.81,
        time_step=time_step,
    )
    system.state = np.array([0.0, 0.0, np.pi, 0.0])
    cost = QuadraticCost(
        running_cost_state_weight=np.diag([5.0, 10.0, 0.1, 0.1]),
        running_cost_control_weight=0.1 * np.identity(1),
        time_step=time_step,
    )
    controller = ModelPredictivePathIntegralController(
        system=system,
        cost=cost,
        time_horizon=100,
        number_of_samples=1000,
        temperature=50.0,
        base_control_confidence=1.0,
        exploration_percentage=0.01,  # Fixme: This is a hack to avoid zero
    )

    cost_trajectory = np.zeros(simulation_time_steps)
    state_trajectory = np.zeros((4, simulation_time_steps))
    time_trajectory = np.zeros(simulation_time_steps)

    time = 0.0
    for k in range(simulation_time_steps):
        time_trajectory[k] = time

        current_state = system.state
        state_trajectory[:, k] = current_state

        controller.reset_systems(current_state)
        control_trajectory = controller.compute_control()
        controller.control_trajectory = np.clip(
            control_trajectory, -100.0, 100.0
        )

        control = controller.control_trajectory[:, 0]
        system.control = control
        time = system.process(time)

        cost.state = current_state
        cost.control = control
        cost_trajectory[k] = cost.evaluate()

        print(
            f"Time: {time}, Control: {control}, State: {current_state}, Cost: {cost_trajectory[k]}"
        )

    # Create the plot
    plot_cart_pole_states(time_trajectory, state_trajectory)


if __name__ == "__main__":
    main()
