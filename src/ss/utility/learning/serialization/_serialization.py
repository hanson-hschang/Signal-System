from types import TracebackType
from typing import (
    Callable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
)

import operator
from collections import defaultdict

import torch

from ss.utility.assertion.inspect import get_nondefault_type_fields
from ss.utility.logging import Logging
from ss.utility.package import (
    get_subclasses,
    import_submodules,
    resolve_module_name,
)

logger = Logging.get_logger(__name__)

SafeCallable: TypeAlias = Union[
    Callable, Tuple[Callable, str], TypeVar, Tuple[TypeVar, str]
]


class SafeCallables(set):
    registered_safe_callables: Set[SafeCallable] = set()
    initialized = False

    def __init__(
        self, safe_callables: Optional[Set[SafeCallable]] = None
    ) -> None:
        if safe_callables is None:
            safe_callables = set()
        super().__init__(safe_callables)

    def __enter__(self) -> None:
        self._list_of_safe_callables: List[SafeCallable] = list(self)
        self._list_of_safe_callables.extend(
            self.__class__.registered_safe_callables
        )
        self._safe_globals_context_manager = torch.serialization.safe_globals(
            self._list_of_safe_callables  # type: ignore
        )
        self._safe_globals_context_manager.__enter__()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self._safe_globals_context_manager.__exit__(
            exc_type, exc_value, traceback
        )

    def to_registered_safe_callables(self) -> None:
        self.__class__.registered_safe_callables.update(self)


def add_subclasses(base_class: Type, package_name: str) -> SafeCallables:
    """
    Register all subclasses of base_class found in the package to be safe callables

    Arguments
    ---------
    base_class: Type
        The base class to find subclasses of
    package_name: str
        The package name to search in (e.g., "ss")
    """

    # Import all submodules to ensure all classes are loaded
    import_submodules(package_name)

    # Get all subclasses
    subclasses = get_subclasses(base_class)

    # Include base class
    all_classes = subclasses.union({base_class})

    # Add all classes and their fields as safe globals
    logger.debug("")
    logger.debug(
        "Add the following classes and their fields as safe globals for torch.load method:"
    )
    safe_callables = SafeCallables()
    for cls in all_classes:
        logger.debug(
            logger.indent() + f"{ cls.__qualname__ } "
            f"( from { resolve_module_name(cls.__module__) } module )"
        )
        unregistered_fields = get_nondefault_type_fields(cls)
        for field_name, field_type in unregistered_fields.items():
            logger.debug(
                logger.indent(2) + f"{ field_name }: "
                f"{ field_type.__qualname__ } "
                f"( from { resolve_module_name(field_type.__module__) } module )"
            )
            # The following lines is a temporary work around to the _get_user_allowed_globals
            # defined in the torch._weights_only_unpickler.py (torch version 2.6.0). Issue link:
            # https://github.com/pytorch/pytorch/issues/146814. The proper way should be
            # module, name = f.__module__, f.__qualname__ rather than the current implementation
            # module, name = f.__module__, f.__name__
            unregistered_type = (
                field_type,
                f"{field_type.__module__}.{field_type.__qualname__}",
            )
            safe_callables.add(unregistered_type)
            # Once the _get_user_allowed_globals is fixed, the following line should be used instead
            # safe_callables.add(field_type)

    # Add all classes to the safe type set
    safe_callables.update(all_classes)

    return safe_callables


def add_type_var() -> SafeCallables:
    from ss.utility.learning.parameter.transformer.config import (
        TC,
    )
    from ss.utility.learning.parameter.transformer.exp.config import (
        TC as TC_EXP,
    )
    from ss.utility.learning.parameter.transformer.softmax.config import (
        TC as TC_SOFTMAX,
    )

    safe_callables = SafeCallables({TC, TC_SOFTMAX, TC_EXP})
    return safe_callables


def add_builtin() -> SafeCallables:
    # This is for getter and setter of properties
    safe_callables = SafeCallables(
        {getattr, setattr, defaultdict, dict, operator.getitem}
    )
    return safe_callables


def add_numpy_types() -> SafeCallables:
    # FIXME: This is a temporary fix to allow for safe loading of numpy scalars
    # and dtypes. This should be removed since there is no numpy dependency in
    # the .pt files currently.

    # Add numpy scalar to safe globals before loading
    from numpy import dtype
    from numpy.dtypes import Float64DType, Int64DType

    safe_callables = SafeCallables({dtype, Int64DType, Float64DType})
    return safe_callables
