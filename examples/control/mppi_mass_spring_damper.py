import click
import matplotlib.pyplot as plt
import numpy as np

from ss.control.cost import CostTrajectoryFigure
from ss.control.cost.quadratic import QuadraticCost
from ss.control.mppi import ModelPredictivePathIntegralController
from ss.system.dense_state.examples.mass_spring_damper import (
    ControlChoice,
    MassSpringDamperStateTrajectoryFigure,
    MassSpringDamperSystem,
)


@click.command()
@click.option(
    "--number-of-connections",
    type=click.IntRange(min=1),
    default=4,
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
    default=0.01,
    help="Set the time step (positive value).",
)
def main(
    number_of_connections: int, simulation_time: float, time_step: float
) -> None:
    simulation_time_steps = int(simulation_time / time_step)
    system = MassSpringDamperSystem(
        number_of_connections=number_of_connections,
        time_step=time_step,
        control_choice=ControlChoice.ALL_FORCES,
    )
    system.state = np.random.rand(2 * number_of_connections)
    cost = QuadraticCost(
        running_cost_state_weight=np.identity(2 * number_of_connections),
        running_cost_control_weight=np.identity(number_of_connections),
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
    state_trajectory = np.zeros(
        (2 * number_of_connections, simulation_time_steps)
    )
    time_trajectory = np.zeros(simulation_time_steps)

    time = 0.0
    for k in range(simulation_time_steps):
        time_trajectory[k] = time

        current_state = system.state
        state_trajectory[:, k] = current_state

        controller.reset_systems(current_state)
        control_trajectory = controller.compute_control()
        control = control_trajectory[:, 0]
        system.control = control
        time = system.process(time)

        cost.state = current_state
        cost.control = control
        cost_trajectory[k] = cost.evaluate()

        print(
            f"Time: {time_trajectory[k]} sec, "
            f"State: {state_trajectory[:, k]}, "
            f"Control: {control}, "
            f"Cost: {cost_trajectory[k]}"
        )

    MassSpringDamperStateTrajectoryFigure(
        time_trajectory,
        state_trajectory,
    ).plot_figure()
    CostTrajectoryFigure(
        time_trajectory,
        cost_trajectory,
    ).plot_figure()
    plt.show()


if __name__ == "__main__":
    main()
