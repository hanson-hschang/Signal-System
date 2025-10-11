from types import TracebackType
from typing import ContextManager

from ss.utility.learning.compile.config import CompileConfig


class CompileContext(ContextManager):
    def __init__(self, compile_config: CompileConfig | None = None) -> None:
        self._compile_config = (
            CompileConfig() if compile_config is None else compile_config
        )

    def __enter__(self) -> None:
        self._previous_compile_config = CompileConfig.get_current()
        self._compile_config.set()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._previous_compile_config.set()
