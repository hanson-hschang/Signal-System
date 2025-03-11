from typing import Final, TypeVar

from dataclasses import dataclass, field

from ss.utility.learning.module.config import BaseLearningConfig


@dataclass
class TransformerConfig(BaseLearningConfig): ...


TC = TypeVar("TC", bound=TransformerConfig)
