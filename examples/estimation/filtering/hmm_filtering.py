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
from ss.utility.logging import Logging
from ss.utility.path import PathManager

logger = Logging.get_logger(__name__)


def hmm_filtering(
    state_dim: int,
    observation_dim: int,
    simulation_time_steps: int,
    step_skip: int,
    number_of_systems: int,
    result_directory: Path,
) -> None:

    def normalize_rows(
        matrix: NDArray,
        temperature: float = 10.0,
    ) -> NDArray:
        matrix = np.exp(temperature * matrix)
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized_matrix: NDArray = matrix / row_sums
        return normalized_matrix

    transition_probability_matrix = normalize_rows(
        np.random.rand(state_dim, state_dim)
    )
    emission_probability_matrix = normalize_rows(
        np.random.rand(state_dim, observation_dim)
    )
    logger.info(f"\n{transition_probability_matrix=}")
    logger.info(f"\n{emission_probability_matrix=}")

    system = HiddenMarkovModel(
        transition_probability_matrix=transition_probability_matrix,
        emission_probability_matrix=emission_probability_matrix,
        number_of_systems=number_of_systems,
    )
    system_callback = MarkovChainCallback(step_skip=step_skip, system=system)

    @njit(cache=True)  # type: ignore
    def observation_model(
        estimated_state: NDArray[np.float64],
        emission_probability_matrix: NDArray[
            np.float64
        ] = system.emission_probability_matrix,
    ) -> NDArray[np.float64]:
        estimated_next_observation = (
            estimated_state @ emission_probability_matrix
        )
        return estimated_next_observation

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

    # Save the data
    system_callback.save(result_directory / "system.hdf5")
    estimator_callback.save(result_directory / "filter.hdf5")

    # Plot the data
    state_trajectory = (
        system_callback["state"]
        if number_of_systems == 1
        else system_callback["state"][0]
    )
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
        state_trajectory=state_trajectory,
        observation_trajectory=observation_trajectory,
        estimated_state_trajectory=estimated_state_trajectory,
        estimated_function_value_trajectory=estimated_function_value_trajectory,
    ).plot()
    plt.show()


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
    "--state-dim",
    type=click.IntRange(min=1),
    default=3,
    help="Set the state dimension (positive integers).",
)
@click.option(
    "--observation-dim",
    type=click.IntRange(min=1),
    default=7,
    help="Set the observation dimension (positive integers).",
)
@click.option(
    "--number-of-systems",
    type=click.IntRange(min=1),
    default=1,
    help="Set the number of systems (positive integers).",
)
@click.option(
    "--random-seed",
    type=click.IntRange(min=0),
    default=2024,
    help="Set the random seed (non-negative integers).",
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
    simulation_time_steps: int,
    step_skip: int,
    state_dim: int,
    observation_dim: int,
    number_of_systems: int,
    random_seed: int,
    verbose: bool,
    debug: bool,
) -> None:

    path_manager = PathManager(__file__)
    Logging.basic_config(
        filename=path_manager.logging_filepath,
        log_level=Logging.Level.DEBUG if debug else Logging.Level.INFO,
        verbose_level=Logging.Level.INFO if verbose else Logging.Level.WARNING,
    )

    np.random.seed(random_seed)

    hmm_filtering(
        state_dim=state_dim,
        observation_dim=observation_dim,
        simulation_time_steps=simulation_time_steps,
        step_skip=step_skip,
        number_of_systems=number_of_systems,
        result_directory=path_manager.result_directory,
    )


if __name__ == "__main__":
    main()
