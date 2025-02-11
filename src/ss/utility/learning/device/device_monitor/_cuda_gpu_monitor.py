from typing import Optional, Union

from pathlib import Path

from ._device_monitor import DeviceMonitor


class CudaGpuMonitor(DeviceMonitor):
    def __init__(
        self,
        sampling_rate: float = 1.0,
        result_directory: Optional[Union[Path, str]] = None,
        result_filename: Optional[str] = None,
    ) -> None:
        super().__init__(
            "CUDA GPU", sampling_rate, result_directory, result_filename
        )
