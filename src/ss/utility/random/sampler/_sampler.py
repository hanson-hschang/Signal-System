from typing import Callable, Tuple, TypeVar, assert_never

import numpy as np
import torch
from numba import njit
from numpy.typing import ArrayLike, NDArray
from scipy.special import softmax

from ss.utility.descriptor import ReadOnlyDescriptor
from ss.utility.map import transform
from ss.utility.random.sampler.config import SamplerConfig


@njit(cache=True)  # type: ignore
def reshape_probability(probability: NDArray) -> Tuple[NDArray, NDArray]:
    sample_shape = np.array(probability.shape[:-1])
    number_of_choices = probability.shape[-1]

    # Reshape probs to 2D: (batch_size, number_of_choices)
    batch_size = int(np.prod(sample_shape)) if sample_shape.size > 0 else 1
    reshaped_probability = probability.reshape(batch_size, number_of_choices)
    return reshaped_probability, sample_shape


def sample(probability: NDArray) -> NDArray:
    # sample_shape = probability.shape[:-1]

    # # Reshape probs to 2D: (batch_size, n_classes)
    # batch_size = np.prod(sample_shape) if sample_shape else 1
    # reshaped_probability = probability.reshape(batch_size, -1)

    reshaped_probability, sample_shape = reshape_probability(probability)

    # Create array to store sampled results
    samples = np.empty(reshaped_probability.shape[0])

    for b, _probability in enumerate(reshaped_probability):
        samples[b] = np.random.choice(len(_probability), p=_probability)

    # Sample for each row in the batch
    # samples = np.array(
    #     [np.random.choice(len(p), p=p) for p in reshaped_probability]
    # )

    # Reshape back to the output shape
    return samples.reshape(sample_shape)


@njit(cache=True)  # type: ignore
def top_k(probability: NDArray, kth: int) -> NDArray:
    # Get the top k probability
    indices = np.argpartition(probability, -kth)[-kth:]
    top_k_probability = np.zeros_like(probability)
    top_k_probability[indices] = probability[indices]
    top_k_probability /= top_k_probability.sum()
    return top_k_probability


@njit(cache=True)  # type: ignore
def top_p(probability: NDArray, p: float) -> NDArray:

    # Sort the probability in descending order
    sorted_indices = np.argsort(probability)[::-1]
    sorted_probability = probability[sorted_indices]

    # Calculate the cumulative probability
    cumulative_probability = np.cumsum(sorted_probability)

    # Find the threshold index
    threshold_index = np.where(cumulative_probability > p)[0][0]

    # Get the top p probability
    indices = sorted_indices[: threshold_index + 1]
    top_p_probability = np.zeros_like(probability)
    top_p_probability[indices] = probability[indices]
    top_p_probability /= top_p_probability.sum()
    return top_p_probability


T = TypeVar("T", NDArray, torch.Tensor)


class Sampler:
    def __init__(self, config: SamplerConfig) -> None:
        self._config = config
        self._temperature = self._config.temperature
        self._sample: Callable[[NDArray], NDArray]
        self._init_sample()

    config = ReadOnlyDescriptor[SamplerConfig]()

    def _init_sample(self) -> None:
        match self._config.option:
            case SamplerConfig.Option.AS_IS:
                self._sample = self._sample_as_is
            case SamplerConfig.Option.TOP_K:
                self._max_number_of_choices = (
                    self._config.max_number_of_choices
                )
                self._sample = self._sample_top_k
            case SamplerConfig.Option.TOP_P:
                self._probability_threshold = (
                    self._config.probability_threshold
                )
                self._sample = self._sample_top_p
            case _ as _option:
                assert_never(_option)

    def to_scaled_probability(self, probability: NDArray) -> NDArray:
        _probability: NDArray = softmax(
            np.log(probability) / self._temperature, axis=-1
        )
        return _probability

    def _sample_as_is(self, probability: NDArray) -> NDArray:
        return sample(probability)

    def _sample_top_k(self, probability: NDArray) -> NDArray:
        number_of_choices = probability.shape[-1]
        max_number_of_choices = (
            min(self._max_number_of_choices, number_of_choices)
            if self._max_number_of_choices > 0
            else number_of_choices
        )
        if max_number_of_choices == number_of_choices:
            return sample(probability)

        # sample_shape = probability.shape[:-1]

        # # Reshape probs to 2D: (batch_size, number_of_choices)
        # batch_size = np.prod(sample_shape) if sample_shape else 1
        # reshaped_probability = probability.reshape(
        #     batch_size, number_of_choices
        # )
        reshaped_probability, sample_shape = reshape_probability(probability)

        # Create array to store sampled results
        samples = np.empty(reshaped_probability.shape[0])

        # Sample for each batch
        for b, _probability in enumerate(reshaped_probability):
            # Get top k probability
            top_k_probability = top_k(_probability, kth=max_number_of_choices)

            # Sample from the top k probability
            samples[b] = np.random.choice(
                number_of_choices, p=top_k_probability
            )

        return samples.reshape(sample_shape)

    def _sample_top_p(self, probability: NDArray) -> NDArray:
        if self._probability_threshold == 1.0:
            return sample(probability)

        number_of_choices = probability.shape[-1]
        # sample_shape = probability.shape[:-1]

        # Reshape probs to 2D: (batch_size, number_of_choices)
        # batch_size = np.prod(sample_shape) if sample_shape else 1
        # reshaped_probability = probability.reshape(
        #     batch_size, number_of_choices
        # )

        reshaped_probability, sample_shape = reshape_probability(probability)

        # Create array to store sampled results
        samples = np.empty(reshaped_probability.shape[0])

        # Sample for each batch
        for b, _probability in enumerate(reshaped_probability):
            # Get top p probability
            top_p_probability = top_p(
                _probability, p=self._probability_threshold
            )

            # Sample from the top p probability
            samples[b] = np.random.choice(
                number_of_choices, p=top_p_probability
            )

        return samples.reshape(sample_shape)

    def sample(self, probability: T) -> T:
        _probability: NDArray
        match probability:
            case torch.Tensor():
                _probability = probability.cpu().numpy()
            case np.ndarray():
                _probability = probability
            case tuple() | list():
                _probability = np.array(probability)
            case _:
                raise TypeError(
                    f"Unsupported type for probability: {type(probability)}"
                )

        sample = self._sample(
            transform(_probability, self.to_scaled_probability)
        )

        match probability:
            case torch.Tensor():
                return torch.tensor(sample).to(device=probability.device)
            case np.ndarray():
                return sample
            case list():
                return sample.tolist()
            case tuple():
                return tuple(sample.tolist())
