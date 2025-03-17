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
from ss.utility.learning.parameter.transformer.norm.min_zero import (
    MinZeroNormTransformer,
)
from ss.utility.learning.parameter.transformer.norm.min_zero.config import (
    MinZeroNormTransformerConfig,
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

        with torch.compiler.set_stance("force_eager"):
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

        with torch.compiler.set_stance("force_eager"):
            with simple_module.evaluation_mode():
                parameter_1: torch.Tensor = simple_module.parameter_1
                result: torch.Tensor = simple_module()
        assert np.allclose(parameter_1.detach().numpy(), np.full((3, 4), 1.0))
        assert np.allclose(result.detach().numpy(), np.full((3, 4), 2.0))


@dataclass
class ComplexConfig(Config.BaseLearningConfig):
    number: ParameterConfig = field(default_factory=ParameterConfig)
    positive_number: PositiveParameterConfig = field(
        default_factory=lambda: PositiveParameterConfig()
    )
    probability_1: ProbabilityParameterConfig = field(
        default_factory=lambda: ProbabilityParameterConfig()
    )
    probability_2: ProbabilityParameterConfig[MinZeroNormTransformerConfig] = (
        field(
            default_factory=lambda: ProbabilityParameterConfig[
                MinZeroNormTransformerConfig
            ](transformer=MinZeroNormTransformerConfig(order=1))
        )
    )


class ComplexModule(BaseLearningModule[ComplexConfig]):
    def __init__(self, config: ComplexConfig) -> None:
        super().__init__(config)
        self._number: Parameter = Parameter(self._config.number, (3, 4))
        self._positive_number: PositiveParameter = PositiveParameter(
            self._config.positive_number, (3, 4)
        )
        self._probability_1: ProbabilityParameter = ProbabilityParameter(
            self._config.probability_1, (3, 4)
        )
        self._probability_2: ProbabilityParameter = ProbabilityParameter(
            self._config.probability_2, (3, 4)
        )

    @property
    def number(self) -> torch.Tensor:
        number: torch.Tensor = self._number()
        return number

    @number.setter
    def number(self, value: torch.Tensor) -> None:
        self._number.set_value(value)

    @property
    def positive_number_parameter(self) -> PositiveParameter:
        return self._positive_number

    @property
    def positive_number(self) -> torch.Tensor:
        positive_number: torch.Tensor = self._positive_number()
        return positive_number

    @positive_number.setter
    def positive_number(self, value: torch.Tensor) -> None:
        self._positive_number.set_value(value)

    @property
    def probability_1_parameter(self) -> ProbabilityParameter:
        return self._probability_1

    @property
    def probability_1(self) -> torch.Tensor:
        probability: torch.Tensor = self._probability_1()
        return probability

    @probability_1.setter
    def probability_1(self, value: torch.Tensor) -> None:
        self._probability_1.set_value(value)

    def forward(self) -> torch.Tensor:
        result: torch.Tensor = torch.mul(
            self._number(), self._probability_1()
        ) + torch.mul(self._positive_number(), self._probability_2())
        return result

    @property
    def probability_2_parameter(
        self,
    ) -> ProbabilityParameter:
        return self._probability_2

    @property
    def probability_2(self) -> torch.Tensor:
        probability: torch.Tensor = self._probability_2()
        return probability

    @probability_2.setter
    def probability_2(self, value: torch.Tensor) -> None:
        self._probability_2.set_value(value)


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
            complex_module.config.positive_number
            == complex_config.positive_number
        )
        assert (
            complex_module.config.probability_1 == complex_config.probability_1
        )

    def test_parameter_initialization(
        self, complex_config: ComplexConfig
    ) -> None:
        complex_module = ComplexModule(complex_config)

        with torch.compiler.set_stance("force_eager"):
            with complex_module.evaluation_mode():
                positive_number = complex_module.positive_number
                probability_1 = complex_module.probability_1
                probability_2 = complex_module.probability_2
        assert positive_number.min() >= 0.0
        assert probability_1.min() >= 0.0
        assert probability_1.max() <= 1.0
        assert probability_2.min() >= 0.0
        assert probability_2.max() <= 1.0
        assert np.allclose(
            np.sum(probability_1.detach().numpy(), axis=-1), np.ones(3)
        )
        assert np.allclose(
            np.sum(probability_2.detach().numpy(), axis=-1), np.ones(3)
        )

    def test_parameter_shape(self, complex_module: ComplexModule) -> None:

        with torch.compiler.set_stance("force_eager"):
            with complex_module.evaluation_mode():
                number = complex_module.number
                positive_number = complex_module.positive_number
                probability_1 = complex_module.probability_1
                probability_2 = complex_module.probability_2
                result: torch.Tensor = complex_module()
        assert number.shape == (3, 4)
        assert positive_number.shape == (3, 4)
        assert probability_1.shape == (3, 4)
        assert probability_2.shape == (3, 4)
        assert result.shape == (3, 4)

    def test_parameter_set_value(self, complex_module: ComplexModule) -> None:
        complex_module.positive_number = torch.full((3, 4), 5.0)
        complex_module.probability_1 = torch.full((3, 4), 0.25)
        complex_module.probability_2 = torch.full((3, 4), 0.25)

        with torch.compiler.set_stance("force_eager"):
            with complex_module.evaluation_mode():
                positive_number = complex_module.positive_number
                probability_1 = complex_module.probability_1
                probability_2 = complex_module.probability_2
        assert np.allclose(
            positive_number.detach().numpy(), np.full((3, 4), 5.0)
        )
        assert np.allclose(
            probability_1.detach().numpy(), np.full((3, 4), 0.25)
        )
        assert np.allclose(
            probability_2.detach().numpy(), np.full((3, 4), 0.25)
        )

    def test_parameter_binding(self, complex_module: ComplexModule) -> None:
        Parameter.binding(
            complex_module.probability_1_parameter,
            complex_module.positive_number_parameter,
        )
        Parameter.binding(
            complex_module.probability_2_parameter,
            complex_module.positive_number_parameter,
        )
        complex_module.positive_number = torch.full((3, 4), torch.e)

        with torch.compiler.set_stance("force_eager"):
            with complex_module.evaluation_mode():
                probability_1 = complex_module.probability_1
                probability_2 = complex_module.probability_2
        assert np.allclose(
            probability_1.detach().numpy(), np.full((3, 4), 0.25)
        )
        assert np.allclose(
            probability_2.detach().numpy(), np.full((3, 4), 0.25)
        )
