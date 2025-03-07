from types import TracebackType
from typing import ContextManager, Dict, Optional, Type

import torch

from ss.utility.device.manager import DeviceManager
from ss.utility.learning import module as Module


class InferenceContext(ContextManager[None]):
    def __init__(self, *modules: Module.BaseLearningModule) -> None:
        self._device_manager = DeviceManager()
        self._modules = tuple(
            self._device_manager.load_module(module) for module in modules
        )
        self._inference_setting: Dict[Module.BaseLearningModule, bool] = dict()
        self._no_grad = torch.no_grad()

    def __enter__(self) -> None:
        self._no_grad.__enter__()

        for module in self._modules:
            self._inference_setting[module] = module.inference
            Module.set_inference_mode(module)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        for module, inference in self._inference_setting.items():
            # module.inference = inference
            Module.set_inference_mode(module, inference)

        self._no_grad.__exit__(exc_type, exc_value, traceback)
