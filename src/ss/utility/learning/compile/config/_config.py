from typing import Optional

from dataclasses import dataclass
from enum import StrEnum, auto

from torch._dynamo import config as dynamo_config
from torch._dynamo import eval_frame


@dataclass
class CompileConfig:

    class Stance(StrEnum):
        DEFAULT = auto()
        FORCE_EAGER = auto()

    stance: Stance = Stance.DEFAULT
    static_shape: bool = False

    @classmethod
    def get_current(cls) -> "CompileConfig":
        compile_config = CompileConfig(
            stance=cls.Stance[eval_frame._stance.stance.upper()],
            static_shape=dynamo_config.force_parameter_static_shapes,
        )
        return compile_config

    def set(self) -> None:
        eval_frame._stance.stance = self.stance.name.lower()
        dynamo_config.force_parameter_static_shapes = self.static_shape
