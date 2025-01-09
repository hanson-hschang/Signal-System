from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

from ss.learning import BaseLearningModule, BaseLearningParameters


@dataclass
class SimpleParameters(BaseLearningParameters):
    input_dim: int = 10
    output_dim: int = 2


# Register the SimpleParameters class with torch.serialization manually
# to allow for safe loading of the model
torch.serialization.add_safe_globals([SimpleParameters])


class SimpleModel(BaseLearningModule):
    def __init__(self, params: SimpleParameters) -> None:
        super().__init__(params)
        self.linear = nn.Linear(params.input_dim, params.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return x


class TestBaseLearningModule:
    @pytest.fixture
    def simple_params(self) -> SimpleParameters:
        return SimpleParameters()

    @pytest.fixture
    def simple_model(self, simple_params: SimpleParameters) -> SimpleModel:
        return SimpleModel(simple_params)

    def test_model_initialization(
        self, simple_params: SimpleParameters
    ) -> None:
        simple_model = SimpleModel(simple_params)
        assert isinstance(simple_model, BaseLearningModule)
        assert simple_model._params == simple_params

    def test_model_save(
        self, simple_model: SimpleModel, tmp_path: Path
    ) -> None:
        simple_model.save(tmp_path / "model.pt")
        assert (tmp_path / "model.pt").exists()

    def test_model_load(
        self, simple_model: SimpleModel, tmp_path: Path
    ) -> None:
        simple_model.save(tmp_path / "model.pt")
        loaded_model = SimpleModel.load(tmp_path / "model.pt")
        assert isinstance(loaded_model, SimpleModel)
        assert loaded_model._params == simple_model._params
        for (
            simple_model_key,
            simple_model_value,
        ), (
            loaded_model_key,
            loaded_model_value,
        ) in zip(
            simple_model.state_dict().items(),
            loaded_model.state_dict().items(),
        ):
            assert simple_model_key == loaded_model_key
            assert torch.equal(simple_model_value, loaded_model_value)
