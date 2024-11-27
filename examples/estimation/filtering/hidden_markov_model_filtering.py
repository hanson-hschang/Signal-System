import os
from pathlib import Path

import click
import matplotlib.pyplot as plt
from tqdm import tqdm

from ss.estimation.estimator import EstimatorCallback
from ss.estimation.filtering.hmm_filtering import (
    HiddenMarkovModelFilter,
    HiddenMarkovModelFilterFigure,
)
from ss.system.finite_state.markov import HiddenMarkovModel, MarkovChainCallback


@click.command()
@click.option(
    "--simulation-time-steps",
    type=click.IntRange(min=1),
    default=100,
    help="Set the simulation time steps (positive integers).",
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
    simulation_time_steps: int,
    step_skip: int,
    number_of_systems: int,
) -> None:
    epsilon = 0.01

    system = HiddenMarkovModel(
        transition_probability_matrix=[
            [0, 0.5, 0.5],
            [epsilon, 1 - epsilon, 0],
            [1 - epsilon, 0, epsilon],
        ],
        emission_probability_matrix=[
            [1, 0],
            [0, 1],
            [0, 1],
        ],
        number_of_systems=number_of_systems,
    )
    system_callback = MarkovChainCallback(step_skip=step_skip, system=system)
    estimator = HiddenMarkovModelFilter(
        system=system,
    )
    estimator_callback = EstimatorCallback(
        step_skip=step_skip,
        estimator=estimator,
    )

    current_time = 0.0
    for k in tqdm(range(simulation_time_steps)):
        system_callback.record(k, current_time)
        estimator_callback.record(k, current_time)

        current_time = system.process(current_time)

        estimator.update(
            observation=system.observe(),
        )
        estimator.estimate()

    system_callback.record(simulation_time_steps, current_time)
    estimator_callback.record(simulation_time_steps, current_time)

    # Save the data
    parent_directory = Path(os.path.dirname(os.path.abspath(__file__)))
    data_folder_directory = parent_directory / Path(__file__).stem
    system_callback.save(data_folder_directory / "system.hdf5")
    estimator_callback.save(data_folder_directory / "filter.hdf5")

    # Plot the data
    observation_trajectory = (
        system_callback["observation"]
        if number_of_systems == 1
        else system_callback["observation"][0]
    )
    estimated_state_trajectory = (
        estimator_callback["estimated_state"]
        if number_of_systems == 1
        else estimator_callback["estimated_state"][0]
    )
    HiddenMarkovModelFilterFigure(
        time_trajectory=estimator_callback["time"],
        observation_trajectory=observation_trajectory,
        estimated_state_trajectory=estimated_state_trajectory,
    ).plot()
    plt.show()


# from matplotlib.colors import Normalize
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# import numpy as np
# fig, ax = plt.subplots()

# # Create a scalar mappable for the colorbar
# norm = Normalize(vmin=0, vmax=1)
# sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
# # sm.set_array([])

# # Create colorbar
# plt.colorbar(sm, ax=ax, orientation='horizontal')

# # Hide the main plot
# ax.set_visible(False)
# plt.show()
# quit()
if __name__ == "__main__":
    main()
