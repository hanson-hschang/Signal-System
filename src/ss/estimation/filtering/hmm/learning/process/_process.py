from typing import Tuple

import torch

from ss.estimation.filtering.hmm.learning.dataset import HmmObservationDataset
from ss.utility.learning.process import BaseLearningProcess
from ss.utility.logging import Logging
from ss.utility.map import transform

logger = Logging.get_logger(__name__)


@torch.compile
def to_log_probability(
    estimation_trajectory: torch.Tensor,
) -> torch.Tensor:
    log_estimation_trajectory = torch.moveaxis(
        torch.log(estimation_trajectory), 1, 2
    )  # (batch_size, estimation_dim, horizon)
    return log_estimation_trajectory


class LearningHmmFilterProcess(BaseLearningProcess):
    def _evaluate_one_batch(
        self, data_batch: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        (
            observation_trajectory,  # (batch_size, max_length)
            target_estimation_trajectory,  # (batch_size, max_length)
        ) = HmmObservationDataset.from_batch(data_batch)
        estimation_trajectory = self._module(
            observation_trajectory=observation_trajectory
        )  # (batch_size, horizon, estimation_dim)
        loss_tensor: torch.Tensor = self._loss_function(
            transform(
                estimation_trajectory, to_log_probability
            ),  # (batch_size, estimation_dim, max_length)
            target_estimation_trajectory,  # (batch_size, max_length)
        )
        return loss_tensor
