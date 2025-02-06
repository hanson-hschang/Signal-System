from typing import Generator, Optional, Type, TypeVar, Union, assert_never

from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path

import torch

from ss.utility.learning.device.device_monitor import (
    CpuMonitor,
    CudaGpuMonitor,
    DeviceMonitor,
    MpsMonitor,
)
from ss.utility.logging import Logging
from ss.utility.singleton import SingletonMeta

logger = Logging.get_logger(__name__)


class Device(StrEnum):
    CUDA_GPU = "cuda"
    CPU = "cpu"
    MPS = "mps"


M = TypeVar("M", bound=torch.nn.Module)


class DeviceManager(metaclass=SingletonMeta):
    def __init__(self, device: Optional[Device] = None) -> None:
        if device is None:
            device = (
                Device.CUDA_GPU if torch.cuda.is_available() else Device.CPU
            )
        match device:
            case Device.CUDA_GPU:
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
            case _ as _device:
                assert_never(_device)
        self._torch_device = torch.device(device)
        self._device = device
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
        sampling_rate: float = 1.0,
        result_directory: Optional[Union[Path, str]] = None,
        result_filename: Optional[str] = None,
    ) -> Generator[DeviceMonitor, None, None]:
        try:
            _DeviceMonitor: Type[DeviceMonitor]
            match self._device:
                case Device.MPS:
                    _DeviceMonitor = MpsMonitor
                case Device.CUDA_GPU:
                    _DeviceMonitor = CudaGpuMonitor
                case Device.CPU:
                    _DeviceMonitor = CpuMonitor
                case _ as _device:
                    assert_never(_device)
            device_monitor = _DeviceMonitor(
                sampling_rate=sampling_rate,
                result_directory=result_directory,
                result_filename=result_filename,
            )
            yield device_monitor.start()
        finally:
            device_monitor.stop()
            device_monitor.save_result()
