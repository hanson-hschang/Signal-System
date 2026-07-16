from typing import Final, Any, TYPE_CHECKING

from importlib import metadata

__version__: Final[str] = metadata.version("Signal-System")

# =============================================================================
# IDE TYPE HINTS (Static Analysis Only)
# =============================================================================
if TYPE_CHECKING:
    pass

# =============================================================================
# RUNTIME LAZY IMPORTS (Dynamic Execution Only)
# =============================================================================
_EXPORTS: dict[str, str] = {}

__all__ = list(_EXPORTS) + ["__version__"]    # type: ignore[reportUnsupportedDunderAll]

def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache, so repeated access doesn't re-resolve
    return value


def __dir__() -> list[str]:
    # Return the list of attributes available in this module, including
    # dynamically imported ones. Override the default behavior of dir()
    # hide the standard attributes like __name__, __doc__, __file__ etc.
    return __all__
