"""Lazy control re-exports -- see ss/__init__.py for rationale."""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ._control import Controller  # noqa: F401
    from .mppi import (  # noqa: F401
        MPPIController,
        MPPIControllerState,
        MPPIDiagnostics,
    )

_EXPORTS: dict[str, str] = {
    "Controller": "_control",
    "MPPIController": "mppi",
    "MPPIControllerState": "mppi",
    "MPPIDiagnostics": "mppi",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return __all__
