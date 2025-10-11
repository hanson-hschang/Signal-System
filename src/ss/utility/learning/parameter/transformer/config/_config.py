from dataclasses import dataclass
from typing import TypeVar

from ss.utility.learning.module.config import BaseLearningConfig


@dataclass
class TransformerConfig(BaseLearningConfig): ...


TC = TypeVar("TC", bound=TransformerConfig)
