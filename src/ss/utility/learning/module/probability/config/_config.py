from dataclasses import dataclass, field

from ss.utility.learning.module.config import BaseLearningConfig
from ss.utility.learning.module.probability.generator.config import (
    ProbabilityGeneratorConfig,
)
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


@dataclass
class ProbabilityConfig(BaseLearningConfig):
    generator: ProbabilityGeneratorConfig = field(
        default_factory=ProbabilityGeneratorConfig
    )
