from typing import Optional, TypeVar, assert_never

from enum import StrEnum

import torch

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
                if torch.cuda.is_available():
                    self._device = torch.device(device)
                else:
                    logger.warning(
                        "No CUDA GPU is available. Switching to CPU."
                    )
                    self._device = torch.device("cpu")
            case Device.CPU:
                self._device = torch.device("cpu")
            case Device.MPS:
                logger.warning(
                    "MPS is not yet supported currently. Switching to CPU."
                )
                self._device = torch.device("cpu")
            case _ as _device:
                assert_never(_device)
        logger.info(f"Device: {self._device}")

    @property
    def device(self) -> torch.device:
        return self._device

    def load_data(self, data: torch.Tensor) -> torch.Tensor:
        return data.to(device=self._device)

    def load_data_batch(
        self, data_batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        return tuple(data.to(device=self._device) for data in data_batch)

    def load_module(self, module: M) -> M:
        return module.to(device=self._device)
