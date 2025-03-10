from typing import TypeVar

from dataclasses import dataclass

from ss.utility.learning.parameter.transformer.config import TransformerConfig


@dataclass
class ExpTransformerConfig(TransformerConfig):
    pass


TC = TypeVar("TC", bound=TransformerConfig, default=ExpTransformerConfig)
