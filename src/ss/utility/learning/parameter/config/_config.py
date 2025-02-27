from dataclasses import dataclass, field

from ss.utility.learning.module.config import BaseLearningConfig
from ss.utility.learning.module.dropout.config import DropoutConfig
from ss.utility.learning.parameter.initializer import InitializerProtocol
from ss.utility.learning.parameter.initializer.normal_distribution import (
    NormalDistributionInitializer,
)


@dataclass
class ParameterConfig(BaseLearningConfig):
    """
    Configuration of the parameter module.
    """

    dropout: DropoutConfig = field(default_factory=DropoutConfig)
    initializer: InitializerProtocol = field(
        default_factory=NormalDistributionInitializer
    )
    require_training: bool = True
