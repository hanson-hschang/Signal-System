from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from ss.estimation.filtering.hmm.learning.module import LearningHmmFilter
from ss.utility.data import Data
from ss.utility.device.manager import DeviceManager
from ss.utility.learning.parameter.transformer.softmax import (
    SoftmaxTransformer,
)
from ss.utility.learning.parameter.transformer.softmax.config import (
    SoftmaxTransformerConfig,
)
from ss.utility.learning.process import BaseLearningProcess
from ss.utility.logging import Logging
from ss.utility.random.sampler import Sampler
from ss.utility.random.sampler.config import SamplerConfig

logger = Logging.get_logger(__name__)


def inference(
    data_filepath: Path,
    model_folderpath: Path,
    model_filename: str,
    result_directory: Path,
) -> None:
    model_filepath = model_folderpath / model_filename

    device_manager = DeviceManager()

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
    learning_filter, _ = LearningHmmFilter[
        SoftmaxTransformer, SoftmaxTransformerConfig
    ].load(
        module_filename,
        safe_callables={
            torch.nn.functional.cross_entropy,
            torch.optim.AdamW,
            # types of extra arguments
        },
    )

    # Set the sampler configuration
    sampler_config = SamplerConfig(temperature=0.5)
    sampler_config.option = SamplerConfig.Option.TOP_P
    sampler_config.probability_threshold = 0.9

    # Prepare the sampler
    sampler = Sampler(sampler_config)

    # Inference
    given_time_horizon = 15
    future_time_steps = 10
    number_of_samples = 5

    logger.info(
        f"The sequence of the first {given_time_horizon} observations from the data is: "
        f"{observation_trajectory[:given_time_horizon]} (given observation)"
    )
    logger.info(
        f"The sequence of the next {future_time_steps} observations from the data is: "
        f"{observation_trajectory[given_time_horizon + 1: given_time_horizon + 1 + future_time_steps]}"
    )
    _observation_trajectory = device_manager.load_data(
        torch.tensor(
            observation_trajectory[:given_time_horizon], dtype=torch.int64
        ).repeat(number_of_samples, 1)
    )

    learning_filter.number_of_systems = number_of_samples
    with BaseLearningProcess.inference_mode(learning_filter):
        learning_filter.update(_observation_trajectory)

        predicted_next_observation_trajectory = torch.empty(
            (number_of_samples, 1, future_time_steps), dtype=torch.int64
        )
        logger.info("")
        logger.info(
            f"Inferring {number_of_samples} samples of sequence of the next {future_time_steps} predictions based on the given observation: "
        )
        for k in logger.progress_bar(range(future_time_steps)):
            predicted_next_observation = torch.from_numpy(
                sampler.sample(learning_filter.estimate().detach().numpy())
            ).to(dtype=predicted_next_observation_trajectory.dtype)
            predicted_next_observation_trajectory[:, 0, k] = (
                predicted_next_observation
            )
            learning_filter.update(predicted_next_observation)

        for i in range(number_of_samples):
            logger.info(
                f"    {predicted_next_observation_trajectory[i, 0, :].numpy()}"
            )
