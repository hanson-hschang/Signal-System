from pathlib import Path

import numpy as np
from tqdm import tqdm

from ss.estimation.filtering.hmm import (
    DualHmmFilter,
    DualHmmFilterCallback,
    HmmFilter,
    HmmFilterCallback,
)
from ss.system.markov import HiddenMarkovModel, HmmCallback
from ss.utility.logging import Logging

from . import figure as Figure
from . import utility as Utility

logger = Logging.get_logger(__name__)


def hmm_filtering(
    state_dim: int,
    discrete_observation_dim: int,
    simulation_time_steps: int,
    step_skip: int,
    batch_size: int,
    result_directory: Path,
) -> None:

    # Create the system parameters
    transition_matrix = Utility.get_probability_matrix(
        state_dim, state_dim, temperature=1 / 3
    )
    emission_matrix = Utility.get_probability_matrix(
        state_dim, discrete_observation_dim, temperature=1 / 6
    )

    # Print the system parameters
    np.set_printoptions(precision=3)
    logger.info("transition_matrix:")
    for row in transition_matrix:
        logger.info(f"    {row}")
    logger.info("emission_matrix:")
    for row in emission_matrix:
        logger.info(f"    {row}")

    # Create the system and the callback
    system = HiddenMarkovModel(
        transition_matrix=transition_matrix,
        emission_matrix=emission_matrix,
        batch_size=batch_size,
    )
    system_callback = HmmCallback(step_skip=step_skip, system=system)

    # Create the filter and the callback
    filter = HmmFilter(
        system=system,
        estimation_matrix=emission_matrix,
    )
    filter_callback = HmmFilterCallback(
        step_skip=step_skip,
        filter=filter,
    )

    # Create the dual filter and the callback
    dual_filter = DualHmmFilter(
        system=system,
        history_horizon=10,
        estimation_matrix=emission_matrix,
    )
    dual_filter_callback = DualHmmFilterCallback(
        step_skip=step_skip,
        filter=dual_filter,
    )

    # Initialization
    current_time = 0.0

    # Compute the initial filter estimation and system observation for dummy values
    system.observe()
    filter.estimate()
    dual_filter.estimate()

    # Record the system and the estimator at the initial time
    system_callback.record(0, current_time)
    filter_callback.record(0, current_time)
    dual_filter_callback.record(0, current_time)

    for k in tqdm(range(1, simulation_time_steps)):

        # Get the observation
        observation = system.observe()

        # Process the system estimation
        filter.estimate()

        # Process the dual filter estimation
        dual_filter.estimate()

        # Update the system
        current_time = system.process(current_time)

        # Update the filter with the observation
        filter.update(observation)

        # Update the dual filter with the observation
        dual_filter.update(observation)

        # Record the system and the estimator
        system_callback.record(k, current_time)
        filter_callback.record(k, current_time)
        dual_filter_callback.record(k, current_time)

    # Save the data
    system_callback.save(result_directory / "system.hdf5")
    filter_callback.save(result_directory / "filter.hdf5")
    dual_filter_callback.save(result_directory / "dual_filter.hdf5")

    # Plot the data
    state_trajectory = (
        system_callback["state"]
        if batch_size == 1
        else system_callback["state"][0]
    )
    observation_trajectory = (
        system_callback["observation"]
        if batch_size == 1
        else system_callback["observation"][0]
    )
    estimated_state_trajectory = (
        filter_callback["estimated_state"]
        if batch_size == 1
        else filter_callback["estimated_state"][0]
    )
    estimation_trajectory = (
        filter_callback["estimation"]
        if batch_size == 1
        else filter_callback["estimation"][0]
    )
    Figure.HmmFilterFigure(
        time_trajectory=filter_callback["time"],
        state_trajectory=state_trajectory,
        observation_trajectory=observation_trajectory,
        estimated_state_trajectory=estimated_state_trajectory,
        estimation_trajectory=estimation_trajectory,
    ).plot()
    Figure.show()
