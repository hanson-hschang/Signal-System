from typing import Callable

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from tool.assertion import isPositiveInteger
from tool.descriptor import ReadOnlyDescriptor


class MovingAveragingSmoother:
    def __init__(self, window_size: int):
        assert isPositiveInteger(
            window_size
        ), f"window_size {window_size} must be a positive integer"
        self._window_size = window_size
        self._init_compute_smoothed_signal()

    window_size = ReadOnlyDescriptor[int]()

    def _init_compute_smoothed_signal(self) -> None:

        self._average_filter = np.ones(self._window_size) / self._window_size

        @njit(cache=True)  # type: ignore
        def compute_smoothed_signal(
            signal: NDArray[np.float64],
            average_filter: NDArray[np.float64] = self._average_filter,
        ) -> NDArray[np.float64]:
            window_size = average_filter.shape[0]

            # Apply convolution
            smoothed_signal = np.convolve(signal, average_filter, mode="same")

            # Handle edge effects by using smaller windows at the borders
            half_window = window_size // 2
            for i in range(half_window):
                # Start of signal
                smoothed_signal[i] = np.mean(signal[: (i + half_window + 1)])
                # End of signal
                smoothed_signal[-(i + 1)] = np.mean(
                    signal[-(i + half_window + 1) :]
                )
            return smoothed_signal

        self._compute_smoothed_signal: Callable[
            [NDArray[np.float64]], NDArray[np.float64]
        ] = compute_smoothed_signal

    def smooth(self, signal: ArrayLike) -> NDArray[np.float64]:
        signal = np.array(signal, dtype=np.float64)
        assert len(signal.shape) == 1, "signal must be a 1D array"
        smoothed_signal: NDArray[np.float64] = self._compute_smoothed_signal(
            signal
        )
        return smoothed_signal
