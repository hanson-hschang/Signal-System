from typing import Any

from pathlib import Path

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

logger = Logging.get_logger(__name__)


def get_estimation_model(
    emission_probability_matrix: NDArray,
) -> Any:
    @njit(cache=True)  # type: ignore
    def estimation_model(
        estimated_state: NDArray,
        emission_probability_matrix: NDArray[
            np.float64
        ] = emission_probability_matrix,
    ) -> NDArray:
        estimation = estimated_state @ emission_probability_matrix
        return estimation

    return estimation_model


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

    estimator = HiddenMarkovModelFilter(
        system=system,
        estimation_model=get_estimation_model(
            emission_probability_matrix=emission_probability_matrix,
        ),
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
    estimation_trajectory = (
        estimator_callback["estimation"]
        if number_of_systems == 1
        else estimator_callback["estimation"][0]
    )
    HiddenMarkovModelFilterFigure(
        time_trajectory=estimator_callback["time"],
        state_trajectory=state_trajectory,
        observation_trajectory=observation_trajectory,
        estimated_state_trajectory=estimated_state_trajectory,
        estimation_trajectory=estimation_trajectory,
    ).plot()
    plt.show()
