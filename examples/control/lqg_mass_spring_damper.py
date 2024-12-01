import os
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are
from tqdm import tqdm

from ss.control.cost import CostCallback, CostTrajectoryFigure
from ss.control.cost.quadratic import QuadraticCost
from ss.control.lqg import LQGController
from ss.system import SystemCallback
from ss.system.examples.mass_spring_damper import (
    ControlChoice,
    MassSpringDamperStateTrajectoryFigure,
    MassSpringDamperSystem,
)


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
def main(
    number_of_connections: int,
    simulation_time: float,
    time_step: float,
    step_skip: int,
    number_of_systems: int,
) -> None:
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
    controller = LQGController(
        system=system,
        cost=cost,
    )

    # # Solve the CARE
    # matrix_P = solve_discrete_are(
    #     a=system._state_space_matrix_A,
    #     b=system._state_space_matrix_B,
    #     q=cost.running_cost_state_weight * time_step,
    #     r=cost.running_cost_control_weight * time_step,
    # )

    current_time = 0.0
    for k in tqdm(range(simulation_time_steps)):

        # Get the current state
        current_state = system.state
        if len(current_state.shape) == 1:
            current_state = current_state[np.newaxis, :]

        # Compute the control for each system
        control = controller.compute_control()

        # Set the control
        system.control = control.squeeze()
        system_callback.record(k, current_time)

        # Compute the cost
        cost.state = current_state.squeeze()
        cost.control = control.squeeze()
        cost_callback.record(k, current_time)

        # Update the system
        current_time = system.process(current_time)

    # Compute the terminal cost
    cost.set_terminal()
    cost.state = system.state
    cost_callback.record(simulation_time_steps, current_time)
    system_callback.record(simulation_time_steps, current_time)

    # Save the data
    parent_directory = Path(os.path.dirname(os.path.abspath(__file__)))
    data_folder_directory = parent_directory / Path(__file__).stem
    system_callback.save(data_folder_directory / "system.hdf5")
    cost_callback.save(data_folder_directory / "cost.hdf5")

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
