from typing import Tuple

import torch

from ss.estimation.filtering.hmm_filtering.learning import config as Config
from ss.estimation.filtering.hmm_filtering.learning._learning_data import (
    HmmObservationDataset,
)
from ss.learning import BaseLearningProcess
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class LearningHmmFilterProcess(BaseLearningProcess):
    def _evaluate_one_batch(
        self, data_batch: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        (
            observation_trajectory,
            next_observation_trajectory,
        ) = HmmObservationDataset.from_batch(
            data_batch
        )  # (batch_size, max_length), (batch_size, max_length)
        predicted_next_observation_log_probability_trajectory = self._model(
            observation_trajectory=observation_trajectory
        )  # (batch_size, discrete_observation_dim, max_length)
        _loss: torch.Tensor = self._loss_function(
            predicted_next_observation_log_probability_trajectory,
            next_observation_trajectory,  # (batch_size, max_length)
        )
        return _loss
