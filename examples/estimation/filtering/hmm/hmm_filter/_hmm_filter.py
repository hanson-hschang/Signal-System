from pathlib import Path

import numpy as np
from tqdm import tqdm

from ss.estimation.dual_filtering.hmm import (
    DualHmmFilter,
    DualHmmFilterCallback,
)
from ss.estimation.filtering.hmm import HmmFilter, HmmFilterCallback
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
    number_of_systems: int,
    result_directory: Path,
) -> None:

    transition_matrix = Utility.get_probability_matrix(
        state_dim, state_dim, temperature=1 / 3
    )
    emission_matrix = Utility.get_probability_matrix(
        state_dim, discrete_observation_dim, temperature=1 / 6
    )

    # transition_matrix = Utility.get_probability_matrix(
    #     state_dim, state_dim, temperature=0.1
    # )
    # emission_matrix = get_probability_matrix(
    #     state_dim, discrete_observation_dim, temperature=6
    # )

    # transition_matrix = np.roll(np.identity(state_dim), 1, axis=0)
    # transition_matrix = np.zeros((state_dim, state_dim))
    # for r in range(state_dim):
    #     values = np.exp(np.random.rand(state_dim)/0.1)
    #     transition_matrix[r, :] = values / np.sum(values)
    # emission_matrix = np.zeros((state_dim, discrete_observation_dim))
    # for r in range(state_dim):
    #     values = np.exp(np.random.rand(discrete_observation_dim))
    #     emission_matrix[r, :] = values / np.sum(values)

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

    filter = HmmFilter(
        system=system,
        estimation_model=Utility.get_estimation_model(
            emission_matrix=np.identity(state_dim),
        ),
    )
    filter_callback = HmmFilterCallback(
        step_skip=step_skip,
        filter=filter,
    )

    max_horizon = 100
    iterations = 2

    dual_filter = DualHmmFilter(
        system=system,
        max_horizon=max_horizon,
    )

    dual_filter_callback = DualHmmFilterCallback(
        step_skip=step_skip,
        filter=dual_filter,
    )

    current_time = 0.0
    estimation_trajectory = np.empty(
        (number_of_systems, state_dim, max_horizon)
    )
    estimation_trajectory[...] = np.nan

    for k in tqdm(range(simulation_time_steps)):
        observation = system.observe()

        # Compute the estimation
        filter.update(observation)
        estimation = filter.estimate()

        # Compute the estimation from dual estimator
        # dual_filter.update(observation)
        # dual_filter.estimate(iterations)
        # result = dual_filter.estimated_distribution_history.copy()
        # result = result[..., 1:]

        # Record the system and the estimator
        system_callback.record(k, current_time)
        filter_callback.record(k, current_time)
        # dual_filter_callback.record(k, current_time)

        # Update the system
        current_time = system.process(current_time)

        # for d in range(state_dim):
        #     estimation_trajectory[d, :] = np.roll(
        #         estimation_trajectory[d, :], -1
        #     )
        # estimation_trajectory[:, :, -1] = estimation

        # if k < max_horizon:
        #     result[:, : -1 - k] = np.nan
        #     # continue

        # time_trajectory = np.arange(k - max_horizon + 1, k + 1)

        # # Plot the data
        # figure = Figure.DualHmmFigure(
        #     time_trajectory=time_trajectory,
        #     estimation_trajectory=estimation_trajectory,
        #     dual_estimation_trajectory=result,
        # ).plot()
        # for t in time_trajectory:
        #     for d in range(state_dim):
        #         figure._subplots[d][0].set_xlim(
        #             time_trajectory[0] - max_horizon / 20,
        #             time_trajectory[-1] + max_horizon / 20,
        #         )
        #         if t % 5 == 0:
        #             figure._subplots[d][0].axvline(
        #                 t, color="black", linewidth=0.5, linestyle="--"
        #             )
        # Figure.show()

    observation = system.observe()

    # Compute the estimation
    filter.update(observation)
    filter.estimate()

    # dual_filter.update(observation)
    # dual_filter.estimate(iterations)

    # Record the system and the estimator
    system_callback.record(simulation_time_steps, current_time)
    filter_callback.record(simulation_time_steps, current_time)
    # dual_filter_callback.record(simulation_time_steps, current_time)

    # Save the data
    system_callback.save(result_directory / "system.hdf5")
    filter_callback.save(result_directory / "filter.hdf5")
    # dual_filter_callback.save(result_directory / "dual_filter.hdf5")

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
        filter_callback["estimated_state"]
        if number_of_systems == 1
        else filter_callback["estimated_state"][0]
    )
    estimation_trajectory = (
        filter_callback["estimation"]
        if number_of_systems == 1
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
