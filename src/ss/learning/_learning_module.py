from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Generator,
    Generic,
    Optional,
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

from ss.learning.config import BLC, BaseLearningConfig
from ss.utility.assertion.validator import FilePathValidator
from ss.utility.learning import serialization
from ss.utility.learning.device import DeviceManager
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


def initialize_safe_callables() -> None:
    if not serialization.SafeCallables.initialized:
        serialization.add_subclasses(
            BaseLearningConfig, "ss"
        ).to_registered_safe_callables()
        serialization.add_builtin().to_registered_safe_callables()
        # Uncomment the following line to register numpy types
        # serialization.add_numpy_types().to_registered_safe_callables()
        serialization.SafeCallables.initialized = True


BLM = TypeVar("BLM", bound="BaseLearningModule")


class BaseLearningModule(nn.Module, Generic[BLC]):
    MODEL_FILE_EXTENSION = (".pt", ".pth")

    def __init__(self, config: BLC) -> None:
        super().__init__()
        assert issubclass(
            type(config), BaseLearningConfig
        ), f"{type(config) = } must be a subclass of {BaseLearningConfig.__name__}"
        self._config = config
        self._inference = False
        self._device_manager = DeviceManager()

    @property
    def config(self) -> BLC:
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
            logger.info("Evaluating model...")
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
        trained_epochs: Optional[int] = None,
    ) -> None:
        filepath = FilePathValidator(
            filename, BaseLearningModule.MODEL_FILE_EXTENSION
        ).get_filepath()
        model_info = dict(
            config=self._config,
            model_state_dict=self.state_dict(),
            trained_epochs=trained_epochs,
        )
        torch.save(model_info, filepath)

    @classmethod
    def load(
        cls: Type[BLM], filename: Union[str, Path]
    ) -> Tuple[BLM, Dict[str, Any]]:
        filepath = FilePathValidator(
            filename, BaseLearningModule.MODEL_FILE_EXTENSION
        ).get_filepath()

        initialize_safe_callables()

        with serialization.SafeCallables():
            model_info: Dict[str, Any] = torch.load(filepath)
            config = cast(BLC, model_info.pop("config"))
            model = cls(config.reload())
            model.load_state_dict(model_info.pop("model_state_dict"))
        return model, model_info


def reset_module(instance: Any) -> None:
    reset_method: Optional[Callable[[], Any]] = getattr(
        instance, "reset", None
    )
    if callable(reset_method):
        reset_method()
