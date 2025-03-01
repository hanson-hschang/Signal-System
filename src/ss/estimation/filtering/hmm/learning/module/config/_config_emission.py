from typing import Sequence, assert_never

from dataclasses import dataclass, field
from enum import StrEnum, auto

# from ss.utility.assertion.validator import (
#     NumberValidator,
#     PositiveNumberValidator,
# )
# from ss.utility.descriptor import ConditionDescriptor
from ss.utility.learning.module.config import BaseLearningConfig

# from ss.utility.learning.module.stochasticizer.config import (
#     StochasticizerConfig,
# )
# from ss.utility.learning.module.probability.config import ProbabilityConfig
# from ss.utility.learning.parameter.initializer import Initializer
from ss.utility.learning.parameter.probability.config import (
    ProbabilityParameterConfig,
)


@dataclass
class EmissionMatrixConfig(BaseLearningConfig):

    probability_parameter: ProbabilityParameterConfig = field(
        default_factory=lambda: ProbabilityParameterConfig()
    )


@dataclass
class EmissionLayerConfig(BaseLearningConfig):

    class Option(StrEnum):
        FULL_MATRIX = auto()

    # @dataclass
    # class InitializerConfig(BaseLearningConfig):

    #     class Option(StrEnum):
    #         NORMAL_DISTRIBUTION = auto()
    #         UNIFORM_DISTRIBUTION = auto()

    #     option: Option = Option.NORMAL_DISTRIBUTION

    #     class MeanDescriptor(
    #         ConditionDescriptor[
    #             float, "EmissionMatrixConfig.InitializerConfig"
    #         ]
    #     ):

    #         def __get__(
    #             self,
    #             obj: "EmissionMatrixConfig.InitializerConfig",
    #             obj_type: type,
    #         ) -> float:
    #             if not (obj.option is obj.Option.NORMAL_DISTRIBUTION):
    #                 raise AttributeError(
    #                     f"'mean' is only available for option = {obj.Option.NORMAL_DISTRIBUTION}. "
    #                     f"The option given is {obj.option}."
    #                 )
    #             return super().__get__(obj, obj_type)

    #         def __set__(
    #             self,
    #             obj: "EmissionMatrixConfig.InitializerConfig",
    #             value: float,
    #         ) -> None:
    #             if not (obj.option is obj.Option.NORMAL_DISTRIBUTION):
    #                 raise AttributeError(
    #                     f"'mean' is only available for option = {obj.Option.NORMAL_DISTRIBUTION}. "
    #                     f"The option given is {obj.option}."
    #                 )
    #             value = NumberValidator(value).get_value()
    #             super().__set__(obj, value)

    #     class StdDescriptor(
    #         ConditionDescriptor[
    #             float, "EmissionMatrixConfig.InitializerConfig"
    #         ]
    #     ):

    #         def __get__(
    #             self,
    #             obj: "EmissionMatrixConfig.InitializerConfig",
    #             obj_type: type,
    #         ) -> float:
    #             if not (obj.option is obj.Option.NORMAL_DISTRIBUTION):
    #                 raise AttributeError(
    #                     f"'variance' is only available for option = {obj.Option.NORMAL_DISTRIBUTION}. "
    #                     f"The option given is {obj.option}."
    #                 )
    #             return super().__get__(obj, obj_type)

    #         def __set__(
    #             self,
    #             obj: "EmissionMatrixConfig.InitializerConfig",
    #             value: float,
    #         ) -> None:
    #             if not (obj.option is obj.Option.NORMAL_DISTRIBUTION):
    #                 raise AttributeError(
    #                     f"'variance' is only available for option = {obj.Option.NORMAL_DISTRIBUTION}. "
    #                     f"The option given is {obj.option}."
    #                 )
    #             value = PositiveNumberValidator(value).get_value()
    #             super().__set__(obj, value)

    #     class MinValueDescriptor(
    #         ConditionDescriptor[
    #             float, "EmissionMatrixConfig.InitializerConfig"
    #         ]
    #     ):

    #         def __get__(
    #             self,
    #             obj: "EmissionMatrixConfig.InitializerConfig",
    #             obj_type: type,
    #         ) -> float:
    #             if not (obj.option is obj.Option.UNIFORM_DISTRIBUTION):
    #                 raise AttributeError(
    #                     f"'min_value' is only available for option = {obj.Option.UNIFORM_DISTRIBUTION}. "
    #                     f"The option given is {obj.option}."
    #                 )
    #             return super().__get__(obj, obj_type)

    #         def __set__(
    #             self,
    #             obj: "EmissionMatrixConfig.InitializerConfig",
    #             value: float,
    #         ) -> None:
    #             if not (obj.option is obj.Option.UNIFORM_DISTRIBUTION):
    #                 raise AttributeError(
    #                     f"'min_value' is only available for option = {obj.Option.UNIFORM_DISTRIBUTION}. "
    #                     f"The option given is {obj.option}."
    #                 )
    #             value = NumberValidator(value).get_value()
    #             super().__set__(obj, value)

    #     class MaxValueDescriptor(
    #         ConditionDescriptor[
    #             float, "EmissionMatrixConfig.InitializerConfig"
    #         ]
    #     ):

    #         def __get__(
    #             self,
    #             obj: "EmissionMatrixConfig.InitializerConfig",
    #             obj_type: type,
    #         ) -> float:
    #             if not (obj.option is obj.Option.UNIFORM_DISTRIBUTION):
    #                 raise AttributeError(
    #                     f"'max_value' is only available for option = {obj.Option.UNIFORM_DISTRIBUTION}. "
    #                     f"The option given is {obj.option}."
    #                 )
    #             return super().__get__(obj, obj_type)

    #         def __set__(
    #             self,
    #             obj: "EmissionMatrixConfig.InitializerConfig",
    #             value: float,
    #         ) -> None:
    #             if not (obj.option is obj.Option.UNIFORM_DISTRIBUTION):
    #                 raise AttributeError(
    #                     f"'max_value' is only available for option = {obj.Option.UNIFORM_DISTRIBUTION}. "
    #                     f"The option given is {obj.option}."
    #                 )
    #             value = NumberValidator(value).get_value()
    #             super().__set__(obj, value)

    #     def __post_init__(self) -> None:
    #         self._mean: float = 0.0
    #         self._std: float = 1.0
    #         self._min_value: float = 0.0
    #         self._max_value: float = 1.0

    #     mean: MeanDescriptor = field(default=MeanDescriptor(), init=False)
    #     std: StdDescriptor = field(default=StdDescriptor(), init=False)
    #     min_value: MinValueDescriptor = field(
    #         default=MinValueDescriptor(), init=False
    #     )
    #     max_value: MaxValueDescriptor = field(
    #         default=MaxValueDescriptor(), init=False
    #     )

    #     def __call__(self, shape: Sequence[int]) -> torch.Tensor:
    #         match self.option:
    #             case self.Option.NORMAL_DISTRIBUTION:
    #                 return Initializer.normal_distribution(
    #                     self._mean, self._std, shape
    #                 )
    #             case self.Option.UNIFORM_DISTRIBUTION:
    #                 return Initializer.uniform_distribution(
    #                     self._min_value, self._max_value, shape
    #                 )
    #             case _ as _invalid_option:
    #                 assert_never(_invalid_option)

    option: Option = Option.FULL_MATRIX
    matrix: EmissionMatrixConfig = field(default_factory=EmissionMatrixConfig)
    # probability_parameter: ProbabilityParameterConfig = field(
    #     default_factory=lambda: ProbabilityParameterConfig()
    # )
    # initializer: InitializerConfig = field(default_factory=InitializerConfig)
    # stochasticizer: StochasticizerConfig = field(
    #     default_factory=StochasticizerConfig
    # )
    # probability: ProbabilityConfig = field(
    #     default_factory=ProbabilityConfig
    # )


@dataclass
class EmissionProcessConfig(BaseLearningConfig):

    layer: EmissionLayerConfig = field(default_factory=EmissionLayerConfig)
