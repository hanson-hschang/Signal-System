import torch

from ss.estimation.filtering.hmm.learning.dataset import HmmObservationDataset
from ss.utility.learning.process import BaseLearningProcess
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


@torch.compile
def to_log_probability(
    estimation_trajectory: torch.Tensor,
) -> torch.Tensor:
    log_estimation_trajectory = torch.log(estimation_trajectory)
    return log_estimation_trajectory


class LearningHmmFilterProcess(BaseLearningProcess):
    def _evaluate_one_batch(
        self, data_batch: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        (
            observation_trajectory,  # (batch_size, observation_dim=1, horizon)
            target_estimation_trajectory,  # (batch_size, observation_dim=1, horizon)  # noqa: E501
        ) = HmmObservationDataset.from_batch(data_batch)
        estimation_trajectory = self._module(
            observation_trajectory=observation_trajectory
        )  # (batch_size, estimation_dim,  horizon)
        loss_tensor: torch.Tensor = self._loss_function(
            torch.log(
                estimation_trajectory
            ),  # (batch_size, estimation_dim, horizon)
            target_estimation_trajectory[:, 0, :],  # (batch_size, horizon)
        )
        return loss_tensor
