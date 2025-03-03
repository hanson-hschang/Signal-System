from typing import cast

from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

from ss.utility.learning.module import BaseLearningModule
from ss.utility.learning.module import config as Config
from ss.utility.learning.parameter import Parameter
from ss.utility.learning.parameter.config import ParameterConfig
from ss.utility.learning.parameter.initializer.normal_distribution import (
    NormalDistributionInitializer,
)
from ss.utility.learning.parameter.initializer.uniform_distribution import (
    UniformDistributionInitializer,
)
from ss.utility.learning.parameter.manifold import ManifoldParameter
from ss.utility.learning.parameter.manifold.config import (
    ManifoldParameterConfig,
)
from ss.utility.learning.parameter.positive import PositiveParameter
from ss.utility.learning.parameter.positive.config import (
    PositiveParameterConfig,
)
from ss.utility.learning.parameter.probability import ProbabilityParameter
from ss.utility.learning.parameter.probability.config import (
    ProbabilityParameterConfig,
)
from ss.utility.learning.parameter.transformer.exp import ExpTransformer
from ss.utility.learning.parameter.transformer.exp.config import (
    ExpTransformerConfig,
)


@dataclass
class SimpleConfig(Config.BaseLearningConfig):
    parameter_1: ParameterConfig = field(default_factory=ParameterConfig)
    parameter_2: ParameterConfig = field(default_factory=ParameterConfig)
    parameter_3: ParameterConfig = field(default_factory=ParameterConfig)


class SimpleModule(BaseLearningModule[SimpleConfig]):
    def __init__(self, config: SimpleConfig) -> None:
        super().__init__(config)
        self._parameter_1 = Parameter(self._config.parameter_1, (3, 4))
        self._parameter_2 = Parameter(self._config.parameter_2, (4, 4))
        self._parameter_3 = Parameter(self._config.parameter_3, (3, 4))

    @property
    def parameter_1(self) -> torch.Tensor:
        parameter: torch.Tensor = self._parameter_1()
        return parameter

    @parameter_1.setter
    def parameter_1(self, value: torch.Tensor) -> None:
        self._parameter_1.set_value(value)

    @property
    def parameter_2(self) -> torch.Tensor:
        parameter: torch.Tensor = self._parameter_2()
        return parameter

    @parameter_2.setter
    def parameter_2(self, value: torch.Tensor) -> None:
        self._parameter_2.set_value(value)

    @property
    def parameter_3(self) -> torch.Tensor:
        parameter: torch.Tensor = self._parameter_3()
        return parameter

    @parameter_3.setter
    def parameter_3(self, value: torch.Tensor) -> None:
        self._parameter_3.set_value(value)

    def forward(self) -> torch.Tensor:
        return torch.matmul(
            self._parameter_1() + self._parameter_3(), self._parameter_2()
        )


class TestParameter:
    @pytest.fixture
    def simple_config(self) -> SimpleConfig:
        return SimpleConfig()

    @pytest.fixture
    def simple_module(self, simple_config: SimpleConfig) -> SimpleModule:
        return SimpleModule(simple_config)

    def test_module_initialization(self, simple_config: SimpleConfig) -> None:
        simple_module = SimpleModule(simple_config)
        assert simple_module.config.parameter_1 == simple_config.parameter_1
        assert simple_module.config.parameter_2 == simple_config.parameter_2

    def test_parameter_initialization(
        self, simple_config: SimpleConfig
    ) -> None:
        simple_config.parameter_1.initializer = (
            NormalDistributionInitializer.basic_config(mean=3.0, std=0.0)
        )
        simple_config.parameter_2.initializer = (
            UniformDistributionInitializer.basic_config(min=1.0, max=2.0)
        )
        simple_module = SimpleModule(simple_config)
        with simple_module.evaluation_mode():
            parameter_1: torch.Tensor = simple_module.parameter_1
            parameter_2: torch.Tensor = simple_module.parameter_2
        assert np.allclose(parameter_1.detach().numpy(), np.full((3, 4), 3.0))
        assert parameter_2.min() >= 1.0
        assert parameter_2.max() <= 2.0

    def test_parameter_shape(self, simple_module: SimpleModule) -> None:
        assert simple_module.parameter_1.shape == (3, 4)
        assert simple_module.parameter_2.shape == (4, 4)

        parameter_1: torch.Tensor = simple_module.parameter_1
        parameter_2: torch.Tensor = simple_module.parameter_2
        result: torch.Tensor = simple_module()
        assert parameter_1.shape == (3, 4)
        assert parameter_2.shape == (4, 4)
        assert result.shape == (3, 4)

    def test_parameter_set_value(self, simple_module: SimpleModule) -> None:
        simple_module.parameter_1 = torch.full((3, 4), 1.0)
        simple_module.parameter_2 = torch.full((4, 4), 2.0)
        with simple_module.evaluation_mode():
            parameter_1: torch.Tensor = simple_module.parameter_1
            parameter_2: torch.Tensor = simple_module.parameter_2
        assert np.allclose(parameter_1.detach().numpy(), np.full((3, 4), 1.0))
        assert np.allclose(parameter_2.detach().numpy(), np.full((4, 4), 2.0))

    def test_parameter_binding(self, simple_module: SimpleModule) -> None:
        Parameter.binding(
            simple_module._parameter_1, simple_module._parameter_3
        )
        simple_module.parameter_3 = torch.full((3, 4), 1.0)
        simple_module.parameter_2 = torch.eye(4)
        with simple_module.evaluation_mode():
            parameter_1: torch.Tensor = simple_module.parameter_1
            result: torch.Tensor = simple_module()
        assert np.allclose(parameter_1.detach().numpy(), np.full((3, 4), 1.0))
        assert np.allclose(result.detach().numpy(), np.full((3, 4), 2.0))


@dataclass
class ComplexConfig(Config.BaseLearningConfig):
    parameter: ParameterConfig = field(default_factory=ParameterConfig)
    positive_parameter: PositiveParameterConfig = field(
        default_factory=lambda: PositiveParameterConfig()
    )
    probability_parameter: ProbabilityParameterConfig = field(
        default_factory=lambda: ProbabilityParameterConfig()
    )


class ComplexModule(BaseLearningModule[ComplexConfig]):
    def __init__(self, config: ComplexConfig) -> None:
        super().__init__(config)
        self._parameter = Parameter[ParameterConfig](
            self._config.parameter, (3, 4)
        )
        self._positive_parameter = PositiveParameter[PositiveParameterConfig](
            self._config.positive_parameter, (3, 4)
        )
        self._probability_parameter = ProbabilityParameter[
            ProbabilityParameterConfig
        ](self._config.probability_parameter, (3, 4))

    @property
    def parameter(self) -> torch.Tensor:
        parameter: torch.Tensor = self._parameter()
        return parameter

    @parameter.setter
    def parameter(self, value: torch.Tensor) -> None:
        self._parameter.set_value(value)

    @property
    def positive_parameter(self) -> torch.Tensor:
        positive_parameter: torch.Tensor = self._positive_parameter()
        return positive_parameter

    @positive_parameter.setter
    def positive_parameter(self, value: torch.Tensor) -> None:
        self._positive_parameter.set_value(value)

    @property
    def probability_parameter(self) -> torch.Tensor:
        probability_parameter: torch.Tensor = self._probability_parameter()
        return probability_parameter

    @probability_parameter.setter
    def probability_parameter(self, value: torch.Tensor) -> None:
        self._probability_parameter.set_value(value)

    def forward(self) -> torch.Tensor:
        result: torch.Tensor = self._parameter() + torch.mul(
            self._positive_parameter(), self._probability_parameter()
        )
        return result


class TestManifoldParameter:
    @pytest.fixture
    def complex_config(self) -> ComplexConfig:
        return ComplexConfig()

    @pytest.fixture
    def complex_module(self, complex_config: ComplexConfig) -> ComplexModule:
        return ComplexModule(complex_config)

    def test_module_initialization(
        self, complex_config: ComplexConfig
    ) -> None:
        complex_module = ComplexModule(complex_config)
        assert (
            complex_module.config.positive_parameter
            == complex_config.positive_parameter
        )
        assert (
            complex_module.config.probability_parameter
            == complex_config.probability_parameter
        )

    def test_parameter_initialization(
        self, complex_config: ComplexConfig
    ) -> None:
        complex_module = ComplexModule(complex_config)
        with complex_module.evaluation_mode():
            positive_parameter = complex_module.positive_parameter
            probability_parameter = complex_module.probability_parameter
        assert positive_parameter.min() >= 0.0
        assert probability_parameter.min() >= 0.0
        assert probability_parameter.max() <= 1.0
        assert np.allclose(
            np.sum(probability_parameter.detach().numpy(), axis=-1), np.ones(3)
        )

    def test_parameter_shape(self, complex_module: ComplexModule) -> None:

        parameter = complex_module.parameter
        positive_parameter = complex_module.positive_parameter
        probability_parameter = complex_module.probability_parameter
        result: torch.Tensor = complex_module()
        assert parameter.shape == (3, 4)
        assert positive_parameter.shape == (3, 4)
        assert probability_parameter.shape == (3, 4)
        assert result.shape == (3, 4)

    def test_parameter_set_value(self, complex_module: ComplexModule) -> None:
        complex_module.positive_parameter = torch.full((3, 4), 5.0)
        complex_module.probability_parameter = torch.full((3, 4), 0.25)
        with complex_module.evaluation_mode():
            positive_parameter = complex_module.positive_parameter
            probability_parameter = complex_module.probability_parameter
        assert np.allclose(
            positive_parameter.detach().numpy(), np.full((3, 4), 5.0)
        )
        assert np.allclose(
            probability_parameter.detach().numpy(), np.full((3, 4), 0.25)
        )

    def test_parameter_binding(self, complex_module: ComplexModule) -> None:
        Parameter.binding(
            complex_module._probability_parameter,
            complex_module._positive_parameter,
        )
        complex_module.positive_parameter = torch.full((3, 4), 1.0)
        with complex_module.evaluation_mode():
            probability_parameter = complex_module.probability_parameter
        assert np.allclose(
            probability_parameter.detach().numpy(), np.full((3, 4), 0.25)
        )
