from typing import TypeVar

from dataclasses import dataclass

from ss.utility.learning.parameter.transformer.config import TransformerConfig


@dataclass
class ExpTransformerConfig(TransformerConfig):
    pass


ExpTC = TypeVar("ExpTC", bound=TransformerConfig, default=ExpTransformerConfig)
