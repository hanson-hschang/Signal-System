from typing import Optional, TypeVar, Union

import time
from pathlib import Path

from ss.utility.assertion.validator import (
    FilePathValidator,
    PositiveNumberValidator,
)
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)

DM = TypeVar("DM", bound="DeviceMonitor")


class DeviceMonitor:
    def __init__(
        self,
        device: Optional[str] = None,
        sampling_rate: float = 1.0,
        result_directory: Optional[Union[Path, str]] = None,
        result_filename: Optional[str] = None,
    ) -> None:
        self._device = device
        self._sampling_rate = PositiveNumberValidator(
            sampling_rate
        ).get_value()
        self._result_directory = (
            result_directory if result_directory else Path.cwd()
        )
        self._result_filename = (
            result_filename
            if result_filename
            else "device_performance_monitoring_result.hdf5"
        )
        result_filepath = Path(self._result_directory) / self._result_filename
        self._result_filepath = FilePathValidator(
            result_filepath, ".hdf5"
        ).get_filepath()

    def start(self: DM) -> DM:
        if self._device is None:
            logger.warning("No monitorable device detected.")
            return self
        logger.debug(
            f"Start performance monitoring on the {self._device} device."
        )
        self._start_time = time.time()
        return self

    def stop(self) -> None:
        self._end_time = time.time()
        logger.debug(
            f"Complete performance monitoring on the {self._device} device."
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
