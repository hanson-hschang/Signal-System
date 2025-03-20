from types import TracebackType
from typing import ContextManager, Optional, Type

from ss.utility.learning.compile.config import CompileConfig


class CompileContext(ContextManager):
    def __init__(self, compile_config: Optional[CompileConfig] = None) -> None:
        self._compile_config = (
            CompileConfig() if compile_config is None else compile_config
        )

    def __enter__(self) -> None:
        self._previous_compile_config = CompileConfig.get_current()
        self._compile_config.set()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self._previous_compile_config.set()
