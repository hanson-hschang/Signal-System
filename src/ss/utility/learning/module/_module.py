from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

import torch
from torch import nn

from ss.utility.learning.step.scheduler import BaseStepScheduler
from ss.utility.assertion.validator import (
    FilePathValidator,
    ReservedKeyNameValidator,
)
from ss.utility.device import Device
from ss.utility.learning import serialization
from ss.utility.learning.module import config as Config
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


def initialize_safe_callables() -> None:
    if not serialization.SafeCallables.initialized:
        from ss.utility.learning.parameter.transformer import Transformer

        serialization.add_config().to_registered_safe_callables()
        serialization.add_subclass(
            Config.BaseLearningConfig, "ss"
        ).to_registered_safe_callables()
        serialization.add_subclass(
            BaseStepScheduler, "ss"
        ).to_registered_safe_callables()
        serialization.add_type_var(
            Transformer, "ss"
        ).to_registered_safe_callables()
        serialization.add_builtin().to_registered_safe_callables()
        # Uncomment the following line to register numpy types
        # serialization.add_numpy_types().to_registered_safe_callables()
        serialization.SafeCallables.initialized = True


BLM = TypeVar("BLM", bound="BaseLearningModule")


class BaseLearningModule(nn.Module, Generic[Config.BLC]):
    FILE_EXTENSIONS = (".pt", ".pth")

    def __init__(self, config: Config.BLC, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert issubclass(type(config), Config.BaseLearningConfig), (
            f"{type(config) = } must be a subclass of "
            f"{Config.BaseLearningConfig.__name__}"
        )
        self._config = config
        self._inference = not self.training
        # self._device_manager = DeviceManager()

    @property
    def config(self) -> Config.BLC:
        return self._config

    def get_parameter(self, name: str) -> nn.Parameter:
        """
        Get training parameter of the module by name.

        Parameters
        ----------
        name : str
            The name of the parameter.

        Returns
        -------
        parameter : nn.Parameter
            The training parameter.
        """
        return super().get_parameter(name)

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
    def training_mode(self) -> Generator[None]:
        try:
            training = self.training
            self.train()
            yield
        finally:
            self.train(training)

    @contextmanager
    def evaluation_mode(self) -> Generator[None]:
        try:
            training = self.training
            self.eval()
            no_grad = torch.no_grad()
            no_grad.__enter__()
            yield
        finally:
            no_grad.__exit__(None, None, None)
            self.train(training)

    def save(
        self,
        filename: str | Path,
        model_info: dict[str, Any] | None = None,
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
        cls: type[BLM],
        filename: str | Path,
        safe_callables: set[serialization.SafeCallable] | None = None,
        strict: bool = True,
    ) -> tuple[BLM, dict[str, Any]]:
        if not strict:
            logger.warning(
                "Loading with strict=False is not recommended. "
                "Do it only if you got the file from a trusted source."
            )
        filepath = FilePathValidator(
            filename, BaseLearningModule.FILE_EXTENSIONS
        ).get_filepath()

        initialize_safe_callables()

        with serialization.SafeCallables(safe_callables):
            model_info: dict[str, Any] = torch.load(
                filepath,
                map_location=Device.CPU,
                weights_only=strict,
            )
            config = cast(Config.BLC, model_info.pop("__config__"))
            module = cls(type(config).reload(config))
            module_state_dict = model_info.pop("__module_state_dict__")
            module.load_state_dict(module_state_dict)

        logger.info("")
        logger.info(f"Module loaded from the file: {filepath.name}")
        return module, model_info


def reset_module(instance: Any) -> None:
    reset_method: Callable[[], Any] | None = getattr(instance, "reset", None)
    if callable(reset_method):
        reset_method()


def set_inference_mode(
    module: BaseLearningModule | nn.Module,
    inference: bool = True,
) -> None:
    if isinstance(module, BaseLearningModule):
        module.inference = inference
    else:
        module.training = not inference
    for child in module.children():
        set_inference_mode(child, inference)
