from typing import Any, Set, Type, TypeVar, Union

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import torch
from torch import nn

from ss.utility.assertion.validator import FilePathValidator


class Mode(StrEnum):
    TRAIN = "TRAIN"
    INFERENCE = "INFERENCE"


@dataclass
class BaseLearningParameters:
    pass


BLM = TypeVar("BLM", bound="BaseLearningModule")


class BaseLearningModule(nn.Module):
    _model_file_extension = (".pt", ".pth")

    def __init__(self, params: BaseLearningParameters) -> None:
        super().__init__()
        assert issubclass(
            type(params), BaseLearningParameters
        ), f"{type(params) = } must be a subclass of {BaseLearningParameters}"
        self._params = params

    def save(self, filename: Union[str, Path], **checkpoint_info: Any) -> None:
        filepath = FilePathValidator(
            filename, self._model_file_extension
        ).get_filepath()
        checkpoint_info["params"] = self._params
        checkpoint_info["model_state_dict"] = self.state_dict()
        torch.save(checkpoint_info, filepath)

    @classmethod
    def load(cls: Type[BLM], filename: Union[str, Path]) -> BLM:
        filepath = FilePathValidator(
            filename, cls._model_file_extension
        ).get_filepath()
        checkpoint_info = torch.load(filepath, weights_only=True)
        model = cls(checkpoint_info["params"])
        model.load_state_dict(checkpoint_info["model_state_dict"])
        return model
