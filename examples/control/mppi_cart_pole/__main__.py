import click
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ss.control.cost import CostCallback, CostTrajectoryFigure
from ss.control.cost.quadratic import QuadraticCost
from ss.control.mppi import ModelPredictivePathIntegralController
from ss.system import SystemCallback
from ss.system.examples.cart_pole import (
    CartPoleStateTrajectoryFigure,
    CartPoleSystem,
)
from ss.utility import basic_config


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
@click.option(
    "--verbose",
    is_flag=True,
    help="Set the verbose mode.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Set the debug mode.",
)
def main(
    simulation_time: float,
    time_step: float,
    step_skip: int,
    verbose: bool,
    debug: bool,
) -> None:
    result_directory = basic_config(__file__, verbose, debug)

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

    current_time = 0.0
    for k in tqdm(range(simulation_time_steps)):
        # Get the current state
        current_state = system.state

        # Compute the control
        controller.system_state = current_state
        controller.compute_control()
        control = controller.control
        # control = np.clip(
        #     control, -100.0, 100.0
        # )

        # Apply the control
        system.control = control
        system_callback.record(k, current_time)

        # Compute the cost
        cost.state = current_state
        cost.control = control
        cost_callback.record(k, current_time)

        # Update the system
        current_time = system.process(current_time)

    # Compute the terminal cost
    cost.set_terminal()
    cost.state = system.state
    cost_callback.record(simulation_time_steps, current_time)
    system_callback.record(simulation_time_steps, current_time)

    # Save the data
    system_callback.save(result_directory / "system.hdf5")
    cost_callback.save(result_directory / "cost.hdf5")

    CartPoleStateTrajectoryFigure(
        system_callback["time"],
        system_callback["state"],
    ).plot()
    CostTrajectoryFigure(
        cost_callback["time"],
        cost_callback["cost"],
    ).plot()
    plt.show()


if __name__ == "__main__":
    main()
