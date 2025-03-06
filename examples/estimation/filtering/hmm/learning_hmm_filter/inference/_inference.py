from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from ss.estimation.filtering.hmm.learning.module import LearningHmmFilter
from ss.utility.data import Data
from ss.utility.device import DeviceManager
from ss.utility.learning.process import BaseLearningProcess
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


def inference(
    data_filepath: Path,
    model_folderpath: Path,
    model_filename: Path,
    result_directory: Path,
) -> None:
    model_filepath = model_folderpath / model_filename

    DeviceManager()

    # Prepare data
    data = Data.load(data_filepath)
    number_of_systems = int(data.meta_info["number_of_systems"])
    observation_trajectory: NDArray = np.array(
        (
            data["observation"][0]
            if number_of_systems == 1
            else data["observation"][0, 0]
        ),
        dtype=np.int64,
    )  # (time_horizon,)

    # Load the module
    module_filename = model_filepath.with_suffix(
        LearningHmmFilter.FILE_EXTENSIONS[0]
    )
    learning_filter, _ = LearningHmmFilter.load(
        module_filename,
        safe_callables={
            torch.nn.functional.cross_entropy,
            torch.optim.AdamW,
            # types of extra arguments
        },
    )

    np.set_printoptions(precision=3, suppress=True)
    with BaseLearningProcess.inference_mode(learning_filter):
        emission_matrix = learning_filter.emission_matrix.numpy()
        logger.info("(layer 0) learned emission matrix = ")
        for k in range(emission_matrix.shape[0]):
            logger.info(f"    {emission_matrix[k]}")
        transition_matrices = [
            [
                _transition_matrix.numpy()
                for _transition_matrix in _transition_layer_matrix
            ]
            for _transition_layer_matrix in learning_filter.transition_matrix
        ]
        initial_states = [
            [_initial_state.numpy() for _initial_state in _initial_state_layer]
            for _initial_state_layer in learning_filter.initial_state
        ]
        for i, transition_layer in enumerate(
            zip(transition_matrices, initial_states), start=1
        ):
            for j, (transition_matrix, initial_state) in enumerate(
                zip(*transition_layer)
            ):
                logger.info(
                    f"(layer {i}, block {j}) learned initial state = {initial_state}"
                )
                logger.info(
                    f"(layer {i}, block {j}) learned transition matrix = "
                )
                for k in range(transition_matrix.shape[0]):
                    logger.info(f"    {transition_matrix[k]}")
                logger.info(
                    "    associated eigenvalues and left eigenvectors:"
                )
                eig_values, eig_vectors = np.linalg.eig(transition_matrix.T)
                for k in range(eig_values.shape[0]):
                    logger.info(
                        f"        {eig_values[k]:+.03f}: {eig_vectors[:, k]}"
                    )

        logger.info("")

    learning_filter.config.prediction.option = (
        learning_filter.config.prediction.Option.TOP_P
    )
    learning_filter.config.prediction.probability_threshold = 0.9
    learning_filter.config.prediction.temperature = 0.5

    # Inference
    given_time_horizon = 20  # This is like prompt length
    future_time_steps = 10  # This is like how many next token to predict
    number_of_samples = 5  # This is like number of answers to generate based on the same prompt (test-time compute)

    with BaseLearningProcess.inference_mode(learning_filter):
        _observation_trajectory = torch.tensor(
            observation_trajectory[:given_time_horizon], dtype=torch.int64
        )
        logger.info(
            f"The sequence of the first {given_time_horizon} observations from the data is: "
            f"{observation_trajectory[:given_time_horizon]} (given observation)"
        )
        logger.info(
            f"The sequence of the next {future_time_steps} observations from the data is: "
            f"{observation_trajectory[given_time_horizon + 1: given_time_horizon + 1 + future_time_steps]}"
        )
        _observation_trajectory = _observation_trajectory.repeat(
            number_of_samples, 1
        )
        learning_filter.update(_observation_trajectory)

        predicted_next_observation_trajectory = torch.empty(
            (number_of_samples, 1, future_time_steps), dtype=torch.int64
        )
        logger.info("")
        logger.info(
            f"Inferring {number_of_samples} samples of sequence of the next {future_time_steps} predictions based on the given observation: "
        )
        for k in logger.progress_bar(range(future_time_steps)):
            predicted_next_observation = learning_filter.predict()
            predicted_next_observation_trajectory[:, :, k] = (
                predicted_next_observation
            )
            learning_filter.update(predicted_next_observation)

        for i in range(number_of_samples):
            logger.info(
                f"    {predicted_next_observation_trajectory[i, 0, :].numpy()}"
            )
