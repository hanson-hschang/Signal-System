from typing import Optional

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ss.tool.assertion import isPositiveInteger
from ss.tool.descriptor import ReadOnlyDescriptor


class MovingAveragingSmoother:
    def __init__(self, window_size: int) -> None:
        assert isPositiveInteger(
            window_size
        ), f"window_size {window_size} must be a positive integer"
        self._window_size = window_size
        self._average_filter = np.ones(self._window_size) / self._window_size

    window_size = ReadOnlyDescriptor[int]()

    def smooth(
        self, signal: ArrayLike, axis: Optional[int] = None
    ) -> NDArray[np.float64]:
        signal = np.array(signal, dtype=np.float64)
        shape = signal.shape
        if len(shape) == 1:
            signal = signal[np.newaxis, :]
            axis = 1
        assert (
            len(signal.shape) == 2
        ), "signal must be a 2D array with the shape (signal_dim, signal_length)."
        if axis is None:
            axis = 1
        assert axis in [0, 1], "axis must be 0 or 1."
        if axis == 0:
            signal = signal.T
        smoothed_signal: NDArray[np.float64] = self._compute_smoothed_signal(
            signal, self._average_filter
        )
        if len(shape) == 1:
            smoothed_signal = smoothed_signal[0]
        if axis == 0:
            smoothed_signal = smoothed_signal.T
        return smoothed_signal

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _compute_smoothed_signal(
        signal: NDArray[np.float64],
        average_filter: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        signal_dim = signal.shape[0]
        window_size = average_filter.shape[0]
        half_window = window_size // 2
        smoothed_signal = np.zeros_like(signal)

        for i in range(signal_dim):
            # Apply convolution
            smoothed_signal[i, :] = np.convolve(
                signal[i, :], average_filter, mode="same"
            )

            # Handle edge effects by using smaller windows at the borders
            for k in range(half_window):
                # Start of signal
                smoothed_signal[i, k] = np.mean(
                    signal[i, : (k + half_window + 1)]
                )
                # End of signal
                smoothed_signal[i, -(k + 1)] = np.mean(
                    signal[i, -(k + half_window + 1) :]
                )
        return smoothed_signal
