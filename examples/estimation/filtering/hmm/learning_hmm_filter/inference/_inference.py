from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from ss.estimation.filtering.hmm.learning.module import LearningHmmFilter
from ss.utility.data import Data
from ss.utility.device import DeviceManager
from ss.utility.learning.mode import LearningMode
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


def inference(
    data_filepath: Path,
    model_filepath: Path,
) -> None:
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

    # Load the model
    learning_filter, _ = LearningHmmFilter.load(
        model_filepath,
        safe_callables={
            torch.nn.functional.cross_entropy,
            torch.optim.AdamW,
            # types of extra arguments
        },
    )
    logger.info(
        f"Load the learning filter model from the file: {model_filepath.name}"
    )
    logger.info("")

    np.set_printoptions(precision=3, suppress=True)
    with LearningMode.inference(learning_filter):
        emission_matrix = learning_filter.emission_matrix.numpy()
        logger.info("(layer 0) learned emission matrix = ")
        for k in range(emission_matrix.shape[0]):
            logger.info(f"    {emission_matrix[k]}")
        transition_matrix_over_layers = [
            transition_matrix.numpy()
            for transition_matrix in learning_filter.transition_matrix
        ]
        for i, transition_matrix in enumerate(
            transition_matrix_over_layers, start=1
        ):
            logger.info(f"(layer {i}) learned transition matrix = ")
            for k in range(transition_matrix.shape[0]):
                logger.info(f"    {transition_matrix[k]}")
            logger.info(
                "    eigenvalues and the corresponded left eigenvectors of the learned transition matrix = "
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

    with LearningMode.inference(learning_filter):
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
