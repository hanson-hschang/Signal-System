from typing import Optional, Union

from pathlib import Path

from torch import mps

from ._device_monitor import DeviceMonitor


class MpsMonitor(DeviceMonitor):
    def __init__(
        self,
        sampling_rate: float = 1.0,
        result_directory: Optional[Union[Path, str]] = None,
        result_filename: Optional[str] = None,
    ) -> None:
        super().__init__(
            "MPS", sampling_rate, result_directory, result_filename
        )
