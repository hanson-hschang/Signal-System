from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch.utils.data import DataLoader

from ss.utility.deprecation import deprecated
from ss.utility.learning import dataset as Dataset


class HmmObservationDataset(Dataset.BaseDataset):
    def __init__(
        self,
        observation: ArrayLike,
        batch_size: int = 1,
        max_length: int = 256,
        stride: int = 64,
    ) -> None:
        observation = np.array(observation)
        if batch_size == 1 and observation.ndim == 2:
            observation = observation[np.newaxis, ...]
        assert observation.ndim == 3, (
            "observation must be a ArrayLike of 3 dimensions "
            "with the shape of (batch_size, observation_dim=1, horizon). "
            f"observation given has the shape of {observation.shape}."
        )
        observation = observation[:, :, :].astype(np.int64)

        time_horizon = observation.shape[-1]
        self._input_trajectory = []
        self._output_trajectory = []

        with torch.no_grad():
            for i in range(batch_size):
                _observation: torch.Tensor = torch.tensor(
                    observation[i], dtype=torch.int64
                )  # (observation_dim=1, horizon,)
                for t in range(0, time_horizon - max_length, stride):
                    input_trajectory: torch.Tensor = _observation[
                        :, t : t + max_length
                    ]  # (max_length,)
                    output_trajectory: torch.Tensor = _observation[
                        :, t + 1 : t + max_length + 1
                    ]  # (max_length,)
                    self._input_trajectory.append(
                        input_trajectory.detach().clone()
                    )
                    self._output_trajectory.append(
                        output_trajectory.detach().clone()
                    )

        self._length = len(self._input_trajectory)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._input_trajectory[index], self._output_trajectory[index]

    @classmethod
    def from_batch(
        cls, batch: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts the data from the batch.

        Parameters
        ----------
        batch: Tuple[torch.Tensor, ...]
            The batch of data.

        Returns
        -------
        input_trajectory: torch.Tensor
            shape: (batch_size, max_length)
            The input_trajectory extracted from the batch.
        output_trajectory: torch.Tensor
            shape: (batch_size, max_length)
            The input_trajectory extracted from the batch.
        """
        input_trajectory, output_trajectory = batch[0], batch[1]
        return input_trajectory, output_trajectory


# @deprecated(
#     alternative_usage="HmmObservationDataset(...).split(...).to_loaders(...)",
# )
# def hmm_observation_data_split_to_loaders(
#     observation: ArrayLike,
#     batch_size: int,
#     max_length: int,
#     stride: int,
#     split_ratio: Sequence[float],
#     batch_size: int = 128,
#     shuffle: bool = True,
#     random_seed: Optional[int] = None,
# ) -> List[DataLoader]:
#     data_loaders = Dataset.dataset_split_to_loaders(
#         dataset=HmmObservationDataset(
#             observation=observation,
#             batch_size=batch_size,
#             max_length=max_length,
#             stride=stride,
#         ),
#         split_ratio=split_ratio,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         random_seed=random_seed,
#     )
#     return data_loaders  # type: ignore
