from typing import Optional, Union

from pathlib import Path

import psutil

from ss.utility.learning.device.device_monitor._device_monitor import (
    DeviceMonitor,
)


class CpuMonitor(DeviceMonitor):
    def __init__(
        self,
        sampling_rate: float = 1.0,
        result_directory: Optional[Union[Path, str]] = None,
        result_filename: Optional[str] = None,
    ) -> None:
        super().__init__(
            "CPU", sampling_rate, result_directory, result_filename
        )
