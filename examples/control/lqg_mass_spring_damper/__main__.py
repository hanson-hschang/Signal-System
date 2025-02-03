import click
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ss.control.cost import CostCallback, CostTrajectoryFigure
from ss.control.cost.quadratic import QuadraticCost
from ss.control.optimal.lqg import LinearQuadraticGaussianController
from ss.system import SystemCallback
from ss.system.examples.mass_spring_damper import (
    ControlChoice,
    MassSpringDamperStateTrajectoryFigure,
    MassSpringDamperSystem,
)
from ss.utility import basic_config


@click.command()
@click.option(
    "--number-of-connections",
    type=click.IntRange(min=1),
    default=2,
    help="Set the number of connections (positive integers).",
)
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
    "--number-of-systems",
    type=click.IntRange(min=1),
    default=1,
    help="Set the number of systems (positive integers).",
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
    number_of_connections: int,
    simulation_time: float,
    time_step: float,
    step_skip: int,
    number_of_systems: int,
    verbose: bool,
    debug: bool,
) -> None:
    path_manager = basic_config(__file__, verbose, debug)

    simulation_time_steps = int(simulation_time / time_step)
    system = MassSpringDamperSystem(
        number_of_connections=number_of_connections,
        time_step=time_step,
        control_choice=ControlChoice.ALL_FORCES,
        process_noise_covariance=0.01 * np.identity(2 * number_of_connections),
        number_of_systems=number_of_systems,
    )
    system.state = np.random.multivariate_normal(
        np.ones(2 * number_of_connections),
        np.identity(2 * number_of_connections),
        size=(number_of_systems,),
    ).squeeze()
    system_callback = SystemCallback(
        step_skip=step_skip,
        system=system,
    )
    cost = QuadraticCost(
        running_cost_state_weight=np.identity(2 * number_of_connections),
        running_cost_control_weight=np.identity(number_of_connections),
        time_step=time_step,
        number_of_systems=number_of_systems,
    )
    cost_callback = CostCallback(
        step_skip=step_skip,
        cost=cost,
    )
    controller = LinearQuadraticGaussianController(
        system=system,
        cost=cost,
    )

    current_time = 0.0
    for k in tqdm(range(simulation_time_steps)):
        # Get the current state
        current_state = system.state

        # Compute the control for each system
        controller.system_state = current_state
        controller.compute_control()
        control = controller.control

        # Set the control
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
    system_callback.save(path_manager.result_directory / "system.hdf5")
    cost_callback.save(path_manager.result_directory / "cost.hdf5")

    # Plot the data
    MassSpringDamperStateTrajectoryFigure(
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
