from pathlib import Path

import numpy as np
from tqdm import tqdm

from ss.estimation.filtering.hmm import HmmFilter, HmmFilterCallback
from ss.system.markov import HiddenMarkovModel, HmmCallback
from ss.utility.logging import Logging

from . import figure as Figure
from .utility import get_estimation_model, get_probability_matrix

logger = Logging.get_logger(__name__)


def hmm_filtering(
    state_dim: int,
    discrete_observation_dim: int,
    simulation_time_steps: int,
    step_skip: int,
    number_of_systems: int,
    result_directory: Path,
) -> None:

    transition_matrix = get_probability_matrix(
        state_dim, state_dim, temperature=3
    )
    emission_matrix = get_probability_matrix(
        state_dim, discrete_observation_dim, temperature=6
    )

    np.set_printoptions(precision=3)
    logger.info("transition_matrix:")
    for row in transition_matrix:
        logger.info(f"    {row}")
    logger.info("emission_matrix:")
    for row in emission_matrix:
        logger.info(f"    {row}")

    system = HiddenMarkovModel(
        transition_matrix=transition_matrix,
        emission_matrix=emission_matrix,
        number_of_systems=number_of_systems,
    )
    system_callback = HmmCallback(step_skip=step_skip, system=system)

    estimator = HmmFilter(
        system=system,
        estimation_model=get_estimation_model(
            emission_matrix=emission_matrix,
        ),
    )
    estimator_callback = HmmFilterCallback(
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
    estimation_trajectory = (
        estimator_callback["estimation"]
        if number_of_systems == 1
        else estimator_callback["estimation"][0]
    )
    Figure.HmmFilterFigure(
        time_trajectory=estimator_callback["time"],
        state_trajectory=state_trajectory,
        observation_trajectory=observation_trajectory,
        estimated_state_trajectory=estimated_state_trajectory,
        estimation_trajectory=estimation_trajectory,
    ).plot()
    Figure.show()
