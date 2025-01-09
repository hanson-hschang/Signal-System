from typing import Any, Generator, List, Sequence, Tuple

import numpy as np
import torch
from matplotlib.axes import Axes
from numba import njit
from numpy.typing import ArrayLike, NDArray
from torch.utils.data import DataLoader, Dataset, random_split

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
        assert observation.ndim == 2, (
            "observation must be a NDArray of 2 dimensions "
            "with the shape of (number_of_systems, time_horizon). "
            f"observation given has the shape of {observation.shape}."
        )

        time_horizon = observation.shape[-1]
        self._input_trajectory = []
        self._output_trajectory = []

        with torch.no_grad():
            for i in range(number_of_systems):
                _observation: torch.Tensor = torch.tensor(
                    observation[i]
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
    observation_trajectory: NDArray,
) -> Generator[Tuple[NDArray, NDArray], None, None]:
    time_horizon = observation_trajectory.shape[-1]
    for k in range(time_horizon - 1):
        yield observation_trajectory[..., k], observation_trajectory[..., k + 1]


@njit(cache=True)  # type: ignore
def cross_entropy(
    input_probability: NDArray,
    target_probability: NDArray,
) -> float:
    return -float(np.mean(target_probability * np.log(input_probability)))


def add_optimal_loss(
    ax: Axes,
    average_loss: float,
) -> None:
    ax.axhline(y=average_loss, color="black", linestyle="--")
    bbox = dict(boxstyle="round", fc="0.8")
    arrowprops = dict(
        arrowstyle="->",
        connectionstyle="angle,angleA=0,angleB=90,rad=10",
    )
    offset = 64
    xlim_min, xlim_max = ax.get_xlim()
    xlim_range = xlim_max - xlim_min
    ax.annotate(
        f"optimal loss: {average_loss:.2f}",
        (xlim_min + 0.1 * xlim_range, average_loss),
        xytext=(2 * offset, offset),
        textcoords="offset pixels",
        bbox=bbox,
        arrowprops=arrowprops,
    )
