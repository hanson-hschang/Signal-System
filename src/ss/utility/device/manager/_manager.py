from typing import Generator, Optional, Type, TypeVar, Union, assert_never

from contextlib import contextmanager
from enum import StrEnum, auto
from pathlib import Path

import torch

from ss.utility.device import Device
from ss.utility.device.monitor import DeviceMonitor
from ss.utility.logging import Logging
from ss.utility.singleton import SingletonMeta

logger = Logging.get_logger(__name__)


M = TypeVar("M", bound=torch.nn.Module)


class DeviceManager(metaclass=SingletonMeta):

    def __init__(self, device: Optional[Device] = None) -> None:
        if device is None:
            device = Device.CUDA if torch.cuda.is_available() else Device.CPU
        match device:
            case Device.CUDA:
                if not torch.cuda.is_available():
                    device = Device.CPU
                    logger.warning(
                        "No CUDA GPU is available. Switching to CPU."
                    )
            case Device.CPU:
                pass
            case Device.MPS:
                device = Device.CPU
                logger.warning(
                    "MPS is not yet supported currently. Switching to CPU."
                )
            case _ as _invalid_device:
                assert_never(_invalid_device)
        self._device = device
        self._torch_device = self._device.torch_device
        logger.info(f"Device: {self._device}")

    @property
    def device(self) -> torch.device:
        return self._torch_device

    def load_data(self, data: torch.Tensor) -> torch.Tensor:
        return data.to(device=self._torch_device)

    def load_data_batch(
        self, data_batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        return tuple(data.to(device=self._torch_device) for data in data_batch)

    def load_module(self, module: M) -> M:
        return module.to(device=self._torch_device)

    @contextmanager
    def monitor_performance(
        self,
        sampling_interval: float = 1.0,
        result_directory: Optional[Union[Path, str]] = None,
        result_filename: Optional[str] = None,
    ) -> Generator[DeviceMonitor, None, None]:
        try:
            device_monitor = DeviceMonitor(
                device=self._device,
                sampling_interval=sampling_interval,
                result_directory=result_directory,
                result_filename=result_filename,
            )
            yield device_monitor.start()
        finally:
            device_monitor.stop()
            device_monitor.save_result()
