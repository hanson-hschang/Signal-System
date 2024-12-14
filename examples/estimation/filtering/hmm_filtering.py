import os
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from numpy.typing import NDArray
from tqdm import tqdm

from ss.estimation.filtering.hmm_filtering import (
    HiddenMarkovModelFilter,
    HiddenMarkovModelFilterCallback,
    HiddenMarkovModelFilterFigure,
)
from ss.system.markov import HiddenMarkovModel, MarkovChainCallback


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

    @njit(cache=True)  # type: ignore
    def observation_model(
        estimated_state: NDArray[np.float64],
        transition_probability_matrix: NDArray[
            np.float64
        ] = system.transition_probability_matrix,
        emission_probability_matrix: NDArray[
            np.float64
        ] = system.emission_probability_matrix,
    ) -> NDArray[np.float64]:
        estimated_observation = (
            estimated_state
            @ transition_probability_matrix
            @ emission_probability_matrix
        )
        return estimated_observation

    estimator = HiddenMarkovModelFilter(
        system=system, estimation_model=observation_model
    )
    estimator_callback = HiddenMarkovModelFilterCallback(
        step_skip=step_skip,
        estimator=estimator,
    )

    current_time = 0.0
    for k in tqdm(range(simulation_time_steps)):

        # Compute the estimation
        estimator.update(observation=system.observe())
        estimator.estimate()

        # Record the system and the estimator
        system_callback.record(k, current_time)
        estimator_callback.record(k, current_time)

        # Update the system
        current_time = system.process(current_time)

    # Compute the estimation
    estimator.update(observation=system.observe())
    estimator.estimate()

    # Record the system and the estimator
    system_callback.record(simulation_time_steps, current_time)
    estimator_callback.record(simulation_time_steps, current_time)

    # Add meta info to callbacks
    meta_info = dict(number_of_systems=number_of_systems)
    system_callback.add_meta_info(meta_info)
    estimator_callback.add_meta_info(meta_info)

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
    estimated_function_value_trajectory = (
        estimator_callback["estimated_function_value"]
        if number_of_systems == 1
        else estimator_callback["estimated_function_value"][0]
    )
    HiddenMarkovModelFilterFigure(
        time_trajectory=estimator_callback["time"],
        observation_trajectory=observation_trajectory,
        estimated_state_trajectory=estimated_state_trajectory,
        estimated_function_value_trajectory=estimated_function_value_trajectory,
    ).plot()
    plt.show()


if __name__ == "__main__":
    main()
