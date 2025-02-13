from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from ss.utility.assertion.validator import FilePathValidator
from ss.utility.device import DeviceManager
from ss.utility.learning import serialization
from ss.utility.logging import Logging

from .. import config as Config

logger = Logging.get_logger(__name__)


def initialize_safe_callables() -> None:
    if not serialization.SafeCallables.initialized:
        serialization.add_subclasses(
            Config.BaseLearningConfig, "ss"
        ).to_registered_safe_callables()
        serialization.add_builtin().to_registered_safe_callables()
        # Uncomment the following line to register numpy types
        # serialization.add_numpy_types().to_registered_safe_callables()
        serialization.SafeCallables.initialized = True


BLM = TypeVar("BLM", bound="BaseLearningModule")


class BaseLearningModule(nn.Module, Generic[Config.BLC]):
    FILE_EXTENSIONS = (".pt", ".pth")

    def __init__(self, config: Config.BLC) -> None:
        super().__init__()
        assert issubclass(
            type(config), Config.BaseLearningConfig
        ), f"{type(config) = } must be a subclass of {Config.BaseLearningConfig.__name__}"
        self._config = config
        self._inference = False
        self._device_manager = DeviceManager()

    @property
    def config(self) -> Config.BLC:
        return self._config

    def reset(self) -> None: ...

    @property
    def inference(self) -> bool:
        return self._inference

    @inference.setter
    def inference(self, inference: bool) -> None:
        self._inference = inference
        self.training = not inference
        for module in self.children():
            if isinstance(module, BaseLearningModule):
                module.inference = inference
            else:
                module.train(self.training)

    @contextmanager
    def training_mode(self) -> Generator[None, None, None]:
        try:
            training = self.training
            self.train()
            yield
        finally:
            self.train(training)

    @contextmanager
    def evaluation_mode(self) -> Generator[None, None, None]:
        try:
            training = self.training
            self.eval()
            yield
        finally:
            self.train(training)

    def save(
        self,
        filename: Union[str, Path],
        **kwargs: Any,
    ) -> None:
        filepath = FilePathValidator(
            filename, BaseLearningModule.FILE_EXTENSIONS
        ).get_filepath()
        module_info = dict(
            config=self._config,
            module_state_dict=self.state_dict(),
            **kwargs,
        )
        torch.save(module_info, filepath)
        logger.debug(f"module saved to {filepath}")

    @classmethod
    def load(
        cls: Type[BLM],
        filename: Union[str, Path],
        safe_callables: Optional[Set[serialization.SafeCallable]] = None,
    ) -> Tuple[BLM, Dict[str, Any]]:
        filepath = FilePathValidator(
            filename, BaseLearningModule.FILE_EXTENSIONS
        ).get_filepath()

        initialize_safe_callables()

        with serialization.SafeCallables(safe_callables):
            module_info: Dict[str, Any] = torch.load(
                filepath,
                map_location=DeviceManager.Device.CPU,
            )
            config = cast(Config.BLC, module_info.pop("config"))
            module = cls(config.reload())
            # model_state_dict is for backward compatibility
            if "model_state_dict" in module_info:
                module_state_dict = module_info.pop("model_state_dict")
            if "module_state_dict" in module_info:
                module_state_dict = module_info.pop("module_state_dict")
            module.load_state_dict(module_state_dict)
        return module, module_info


def reset_module(instance: Any) -> None:
    reset_method: Optional[Callable[[], Any]] = getattr(
        instance, "reset", None
    )
    if callable(reset_method):
        reset_method()
