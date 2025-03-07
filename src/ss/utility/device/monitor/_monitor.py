from typing import Optional, Self, Union, assert_never

import threading
import time
from pathlib import Path

from ss.utility.assertion.validator import (
    FilePathValidator,
    PositiveNumberValidator,
)
from ss.utility.device import Device
from ss.utility.device.performance import (
    Performance,
    PerformanceCallback,
)
from ss.utility.device.performance.cpu import CpuPerformance
from ss.utility.device.performance.cuda import CudaGpuPerformance
from ss.utility.device.performance.mps import MpsPerformance
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class DeviceMonitor:
    def __init__(
        self,
        device: Device,
        sampling_interval: float = 1.0,
        result_directory: Optional[Union[Path, str]] = None,
        result_filename: Optional[str] = None,
    ) -> None:
        self._device = device
        self._sampling_interval = PositiveNumberValidator(
            sampling_interval
        ).get_value()
        self._result_directory = (
            result_directory if result_directory else Path.cwd()
        )
        self._result_filename = (
            result_filename if result_filename else "device_performance.hdf5"
        )
        result_filepath = Path(self._result_directory) / self._result_filename
        self._result_filepath = FilePathValidator(
            result_filepath, ".hdf5"
        ).get_filepath()

        self._start_time = 0.0
        self._end_time = 0.0
        self._thread: Optional[threading.Thread] = None
        self._event = threading.Event()

        self._performance: Performance
        match self._device:
            case Device.CPU:
                self._performance = CpuPerformance()
            case Device.CUDA:
                self._performance = CudaGpuPerformance()
            case Device.MPS:
                self._performance = MpsPerformance()
            case _ as _invalid_device:
                assert_never(_invalid_device)
        self._performance_callback = PerformanceCallback(self._performance)

    @property
    def time(self) -> float:
        return time.time() - self._start_time

    def loop(self) -> None:
        current_step = 0
        while not self._event.is_set():
            try:
                self._performance_callback.record(current_step, self.time)
                current_step = self._performance.process(current_step)
                time.sleep(self._sampling_interval)
            except Exception as e:
                logger.error(f"Error in monitoring: {e}")
                time.sleep(self._sampling_interval)

    def start(self) -> Self:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self.loop, daemon=True)
        self._event.clear()
        logger.debug(
            f"Start performance monitoring on the {self._device} device."
        )
        self._start_time = time.time()
        self._thread.start()
        return self

    def stop(self) -> None:
        self._end_time = self.time
        if self._thread and self._thread.is_alive():
            self._event.set()
            self._thread.join(timeout=self._sampling_interval * 2)
            self._thread = None
        logger.debug(
            f"Complete performance monitoring on the {self._device} device in {self._end_time:.2f} seconds."
        )

    def save_result(self) -> None:
        if self._device is None:
            logger.warning(
                "No result file is saved because of no monitorable device detected."
            )
            return
        logger.debug(
            f"Save the {self._device} device performance monitoring result to {self._result_filepath}."
        )
        self._performance_callback.save(self._result_filepath)
