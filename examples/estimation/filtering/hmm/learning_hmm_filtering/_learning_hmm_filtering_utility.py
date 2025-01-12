from typing import Any, Generator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from numba import njit
from numpy.typing import ArrayLike, NDArray
from torch.utils.data import DataLoader, Dataset, random_split

from ss.system.markov import one_hot_encoding
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class ObservationDataset(Dataset):
    def __init__(
        self,
        observation: ArrayLike,
        number_of_systems: int = 1,
        max_length: int = 256,
        stride: int = 64,
    ) -> None:
        observation = np.array(observation)
        if number_of_systems == 1:
            observation = observation[np.newaxis, ...]
        assert observation.ndim == 3, (
            "observation must be a NDArray of 3 dimensions "
            "with the shape of (number_of_systems, 1, time_horizon). "
            f"observation given has the shape of {observation.shape}."
        )
        observation = observation[:, 0, ...].astype(np.int64)

        time_horizon = observation.shape[-1]
        self._input_trajectory = []
        self._output_trajectory = []

        with torch.no_grad():
            for i in range(number_of_systems):
                _observation: torch.Tensor = torch.tensor(
                    observation[i], dtype=torch.int64
                )  # (time_horizon,)
                for t in range(0, time_horizon - max_length, stride):
                    input_trajectory: torch.Tensor = _observation[
                        t : t + max_length
                    ]  # (max_length,)
                    output_trajectory: torch.Tensor = _observation[
                        t + 1 : t + max_length + 1
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
    def from_batch(cls, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        # The indexing of batch is defined by the __getitem__ method.
        input_trajectory, output_trajectory = batch[0], batch[1]
        return input_trajectory, output_trajectory


def data_split(
    observation: ArrayLike,
    split_ratio: Sequence[float],
    number_of_systems: int = 1,
    batch_size: int = 128,
    max_length: int = 256,
    stride: int = 64,
    random_seed: int = 2025,
) -> List[DataLoader]:
    generator = torch.Generator().manual_seed(random_seed)
    datasets = random_split(
        dataset=ObservationDataset(
            observation=observation,
            number_of_systems=number_of_systems,
            max_length=max_length,
            stride=stride,
        ),
        lengths=split_ratio,
        generator=generator,
    )
    dataloaders = []
    for dataset in datasets:
        dataloaders.append(
            DataLoader(dataset, batch_size=batch_size, shuffle=True)
        )
    return dataloaders


def get_observation_model(
    transition_probability_matrix: NDArray,
    emission_probability_matrix: NDArray,
    future_time_steps: int = 0,
) -> Any:
    @njit(cache=True)  # type: ignore
    def observation_model(
        estimated_state: NDArray,
        transition_probability_matrix: NDArray = transition_probability_matrix,
        emission_probability_matrix: NDArray = emission_probability_matrix,
        future_time_steps: int = future_time_steps,
    ) -> NDArray:
        for _ in range(future_time_steps):
            estimated_state = estimated_state @ transition_probability_matrix
        estimated_next_observation: NDArray = (
            estimated_state @ emission_probability_matrix
        )
        return estimated_next_observation

    return observation_model


def observation_generator(
    observation_trajectory: NDArray[np.int64],
    discrete_observation_dim: Optional[int] = None,
) -> Generator[Tuple[NDArray, NDArray], None, None]:
    time_horizon = observation_trajectory.shape[-1]
    if discrete_observation_dim is None:
        discrete_observation_dim = int(np.max(observation_trajectory)) + 1
    observation_encoder_basis = np.identity(
        discrete_observation_dim, dtype=np.float64
    )
    for k in range(time_horizon - 1):
        yield (
            observation_trajectory[..., k],
            one_hot_encoding(
                observation_trajectory[:, 0, k + 1],
                observation_encoder_basis,
            ),
        )


@njit(cache=True)  # type: ignore
def cross_entropy(
    input_probability: NDArray,
    target_probability: NDArray,
) -> float:
    """
    Compute the batch size cross-entropy loss, which is defined as
    :math: `-\\frac{1}{N} \\sum_{i=1}^{N} \\sum_{j=1}^{C} p^*_{ij} \\log(p_{ij})`
    where :math: `N` is the batch size, :math: `C` is the number of classes,
    :math: `p^*_{ij}` is the `target_probability`, and :math: `p_{ij}` is the `input_probability`.

    Parameters
    ----------
    input_probability : NDArray
        probability of input with shape (num_classes,) or (batch_size, num_classes)
        values should be in the range of :math: `(0, 1]`
    target_probability : NDArray
        probability of target with the same shape as input_probability
        values should be in the range of :math: `[0, 1]`

    Returns
    -------
    loss : float
        The cross-entropy loss of the input and target probability.
    """

    num_classes = input_probability.shape[1]
    loss = (
        -np.mean(target_probability * np.log(input_probability)) * num_classes
    )
    return float(loss)
