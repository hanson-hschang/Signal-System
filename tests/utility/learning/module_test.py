from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

from ss.utility.learning.module import BaseLearningModule
from ss.utility.learning.module import config as Config


@dataclass
class SimpleConfig(Config.BaseLearningConfig):
    input_dim: int = 10
    output_dim: int = 2


class SimpleModule(BaseLearningModule[SimpleConfig]):
    def __init__(self, config: SimpleConfig) -> None:
        super().__init__(config)
        self.linear = nn.Linear(
            self._config.input_dim, self._config.output_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return x


class TestBaseLearningModule:
    @pytest.fixture
    def simple_config(self) -> SimpleConfig:
        return SimpleConfig()

    @pytest.fixture
    def simple_module(self, simple_config: SimpleConfig) -> SimpleModule:
        return SimpleModule(simple_config)

    def test_module_initialization(self, simple_config: SimpleConfig) -> None:
        simple_module = SimpleModule(simple_config)
        assert isinstance(simple_module, BaseLearningModule)
        assert simple_module.config == simple_config

    def test_module_save(
        self, simple_module: SimpleModule, tmp_path: Path
    ) -> None:
        simple_module.save(tmp_path / "model.pt")
        assert (tmp_path / "model.pt").exists()

    def test_module_load(
        self, simple_module: SimpleModule, tmp_path: Path
    ) -> None:
        simple_module.save(tmp_path / "model.pt", dict(trained_epochs=10))
        loaded_module, loaded_model_info = SimpleModule.load(
            tmp_path / "model.pt",
        )
        assert isinstance(loaded_module, SimpleModule)
        assert loaded_module._config == simple_module._config
        for (
            simple_module_key,
            simple_module_value,
        ), (
            loaded_module_key,
            loaded_module_value,
        ) in zip(
            simple_module.state_dict().items(),
            loaded_module.state_dict().items(),
        ):
            assert simple_module_key == loaded_module_key
            assert torch.equal(simple_module_value, loaded_module_value)
        assert loaded_model_info["trained_epochs"] == 10
