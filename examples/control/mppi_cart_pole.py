import click
import matplotlib.pyplot as plt
import numpy as np

from ss.control.cost import CostCallback, CostTrajectoryFigure
from ss.control.cost.quadratic import QuadraticCost
from ss.control.mppi import ModelPredictivePathIntegralController
from ss.system.state_vector import SystemCallback
from ss.system.state_vector.examples.cart_pole import (
    CartPoleStateTrajectoryFigure,
    CartPoleSystem,
)


@click.command()
@click.option(
    "--simulation-time",
    type=click.FloatRange(min=0),
    default=1.0,
    help="Set the simulation time (positive value).",
)
@click.option(
    "--time-step",
    type=click.FloatRange(min=0),
    default=0.02,
    help="Set the time step (positive value).",
)
@click.option(
    "--step-skip",
    type=click.IntRange(min=1),
    default=1,
    help="Set the step skip (positive integers).",
)
def main(
    simulation_time: float,
    time_step: float,
    step_skip: int,
) -> None:
    simulation_time_steps = int(simulation_time / time_step)
    system = CartPoleSystem(
        cart_mass=1.0,
        pole_mass=0.01,
        pole_length=2.0,
        gravity=9.81,
        time_step=time_step,
    )
    system.state = np.array([0.0, 0.0, np.pi, 0.0])
    system_callback = SystemCallback(
        step_skip=step_skip,
        system=system,
    )
    cost = QuadraticCost(
        running_cost_state_weight=np.diag([5.0, 10.0, 0.1, 0.1]),
        running_cost_control_weight=0.1 * np.identity(1),
        time_step=time_step,
    )
    cost_callback = CostCallback(
        step_skip=step_skip,
        cost=cost,
    )
    controller = ModelPredictivePathIntegralController(
        system=system,
        cost=cost,
        time_horizon=100,
        number_of_samples=1000,
        temperature=100.0,
        base_control_confidence=0.98,
    )

    time = 0.0
    for k in range(simulation_time_steps):

        # Get the current state
        current_state = system.state

        # Compute the control
        controller.reset_systems(current_state)
        control_trajectory = controller.compute_control()
        # controller.control_trajectory = np.clip(
        #     control_trajectory, -100.0, 100.0
        # )
        control = controller.control_trajectory[:, 0]

        # Apply the control
        system.control = control
        system_callback.make_callback(k, time)
        time = system.process(time)

        # Compute the cost
        cost.state = current_state
        cost.control = control
        cost_callback.make_callback(k, time)

    # Compute the terminal cost
    cost.set_terminal()
    cost_callback.make_callback(simulation_time_steps, time)

    CartPoleStateTrajectoryFigure(
        system_callback["time"],
        system_callback["state"],
    ).plot_figure()
    CostTrajectoryFigure(
        cost_callback["time"],
        cost_callback["cost"],
    ).plot_figure()
    plt.show()


if __name__ == "__main__":
    main()
