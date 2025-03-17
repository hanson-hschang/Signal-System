from typing import Callable, assert_never

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import softmax

from ss.utility.descriptor import ReadOnlyDescriptor
from ss.utility.map import transform
from ss.utility.random.sampler.config import SamplerConfig


def sample(probability: NDArray) -> NDArray:
    sample_shape = probability.shape[:-1]

    # Reshape probs to 2D: (batch_size, n_classes)
    batch_size = np.prod(sample_shape) if sample_shape else 1
    reshaped_probability = probability.reshape(batch_size, -1)

    # Sample for each row in the batch
    samples = np.array(
        [np.random.choice(len(p), p=p) for p in reshaped_probability]
    )

    # Reshape back to the output shape
    return samples.reshape(sample_shape)


def top_k(probability: NDArray, kth: int) -> NDArray:
    # Get the top k probability
    indices = np.argpartition(probability, -kth)[-kth:]
    top_k_probability = np.zeros_like(probability)
    top_k_probability[indices] = probability[indices]
    top_k_probability /= top_k_probability.sum()
    return top_k_probability


def top_p(probability: NDArray, p: float) -> NDArray:

    # Sort the probability in descending order
    sorted_indices = np.argsort(probability)[::-1]
    sorted_probability = probability[sorted_indices]

    # Calculate the cumulative probability
    cumulative_probability = np.cumsum(sorted_probability)

    # Find the threshold index
    threshold_index = np.where(cumulative_probability > p)[0][0]

    # Get the top p probability
    indices = sorted_indices[:threshold_index]
    top_p_probability = np.zeros_like(probability)
    top_p_probability[indices] = probability[indices]
    top_p_probability /= top_p_probability.sum()
    return top_p_probability


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

        sample_shape = probability.shape[:-1]

        # Reshape probs to 2D: (batch_size, number_of_choices)
        batch_size = np.prod(sample_shape) if sample_shape else 1
        reshaped_probability = probability.reshape(
            batch_size, number_of_choices
        )

        # Create array to store sampled results
        samples = np.empty(batch_size)

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
        sample_shape = probability.shape[:-1]

        # Reshape probs to 2D: (batch_size, number_of_choices)
        batch_size = np.prod(sample_shape) if sample_shape else 1
        reshaped_probability = probability.reshape(
            batch_size, number_of_choices
        )

        # Create array to store sampled results
        samples = np.empty(batch_size)

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

    def sample(self, probability: ArrayLike) -> NDArray:
        _probability = transform(
            np.array(probability), self.to_scaled_probability
        )
        sample = self._sample(_probability)
        return sample
