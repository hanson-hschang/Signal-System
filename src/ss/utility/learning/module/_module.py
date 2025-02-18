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

from ss.utility.assertion.validator import (
    FilePathValidator,
    ReservedKeyNameValidator,
)
from ss.utility.device import DeviceManager
from ss.utility.learning import serialization
from ss.utility.learning.module import config as Config
from ss.utility.logging import Logging

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
        self._inference = not self.training
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
        # for module in self.children():
        #     print(self.__class__.__name__, module.__class__.__name__)
        #     if isinstance(module, BaseLearningModule):
        #         module.inference = inference
        #     else:
        #         module.train(self.training)
        #         print([c.__class__.__name__ for c in module.children()])

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
        model_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        if model_info is None:
            model_info = dict()
        module_info = dict(
            __config__=self._config,
            __module_state_dict__=self.state_dict(),
        )
        ReservedKeyNameValidator(
            model_info, module_info.keys(), allow_dunder_names=True
        )
        filepath = FilePathValidator(
            filename, BaseLearningModule.FILE_EXTENSIONS
        ).get_filepath()
        model_info.update(module_info)
        torch.save(model_info, filepath)
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
            model_info: Dict[str, Any] = torch.load(
                filepath,
                map_location=DeviceManager.Device.CPU,
            )
            # config is for backward compatibility
            # if "config" in model_info:
            #     config = cast(Config.BLC, model_info.pop("config"))
            # if "__config__" in model_info:
            config = cast(Config.BLC, model_info.pop("__config__"))
            module = cls(config.reload())
            # model_state_dict and module_state_dict are for backward compatibility
            # if "model_state_dict" in model_info:
            #     module_state_dict = model_info.pop("model_state_dict")
            # if "module_state_dict" in model_info:
            #     module_state_dict = model_info.pop("module_state_dict")
            # if "__module_state_dict__" in model_info:
            module_state_dict = model_info.pop("__module_state_dict__")
            module.load_state_dict(module_state_dict)
            logger.info(f"Load the module from the file: {filepath.name}")
            logger.info("")
        return module, model_info


def reset_module(instance: Any) -> None:
    reset_method: Optional[Callable[[], Any]] = getattr(
        instance, "reset", None
    )
    if callable(reset_method):
        reset_method()


def set_inference_mode(
    module: Union[BaseLearningModule, nn.Module],
    inference: bool = True,
) -> None:
    if isinstance(module, BaseLearningModule):
        module.inference = inference
    else:
        module.training = not inference
    for child in module.children():
        set_inference_mode(child, inference)
