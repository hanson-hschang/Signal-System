"""Lazy cost re-exports."""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ._cost import Cost
    from ._cost_quadratic import QuadraticCost

_EXPORTS: dict[str, str] = {
    "Cost": "_cost",
    "QuadraticCost": "_cost_quadratic",
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
