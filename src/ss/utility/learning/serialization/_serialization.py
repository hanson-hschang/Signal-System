from collections.abc import Callable
from types import TracebackType
from typing import TypeAlias, TypeVar, Union

import torch

from ss.utility.assertion.inspect import get_nondefault_type_fields
from ss.utility.logging import Logging
from ss.utility.package import Package

logger = Logging.get_logger(__name__)

SafeCallable: TypeAlias = Union[
    Callable, tuple[Callable, str], TypeVar, tuple[TypeVar, str]
]


class SafeCallables(set):
    registered_safe_callables: set[SafeCallable] = set()
    initialized = False

    def __init__(
        self, safe_callables: set[SafeCallable] | None = None
    ) -> None:
        if safe_callables is None:
            safe_callables = set()
        super().__init__(safe_callables)

    def __enter__(self) -> None:
        self._list_of_safe_callables: list[SafeCallable] = list(self)
        self._list_of_safe_callables.extend(
            self.__class__.registered_safe_callables
        )
        self._safe_globals_context_manager = torch.serialization.safe_globals(
            self._list_of_safe_callables  # type: ignore
        )
        self._safe_globals_context_manager.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._safe_globals_context_manager.__exit__(
            exc_type, exc_value, traceback
        )

    def to_registered_safe_callables(self) -> None:
        self.__class__.registered_safe_callables.update(self)


def add_config() -> SafeCallables:
    from ss.utility.learning.process.config import ProcessConfig
    from ss.utility.random.sampler.config import SamplerConfig

    safe_callables = SafeCallables(
        {
            ProcessConfig,
            ProcessConfig.Mode,
            SamplerConfig,
            SamplerConfig.Option,
        }
    )
    return safe_callables


def add_sub_dataclass(
    base_dataclass: type, package_name: str
) -> SafeCallables:
    """
    Register all subclasses of base_dataclass found in package
    as safe callables

    Arguments
    ---------
    base_dataclass: Type
        The base dataclass to find subclasses of
    package_name: str
        The package name to search in (e.g., "ss")
    """

    # Import all submodules to ensure all classes are loaded
    Package.import_submodules(package_name)

    # Get all subclasses
    subclasses = Package.get_subclasses(base_dataclass)

    # Include base class
    all_classes = subclasses.union({base_dataclass})

    # Add all classes and their fields as safe globals
    logger.debug("")
    logger.debug(
        "Add the following classes and their fields "
        "as safe globals for torch.load method:"
    )
    safe_callables = SafeCallables()
    for cls in all_classes:
        logger.debug(
            f"{cls.__qualname__} "
            f"( from {Package.resolve_module_name(cls.__module__)} module )",
            indent_level=1,
        )
        unregistered_fields = get_nondefault_type_fields(cls)
        for field_name, field_type in unregistered_fields.items():
            logger.debug(
                f"{field_name}: "
                f"{field_type.__qualname__} (from "
                f"{Package.resolve_module_name(field_type.__module__)}"
                " module)",
                indent_level=2,
            )
            # The following lines is a temporary work around to
            # the _get_user_allowed_globals defined in the
            # torch._weights_only_unpickler.py (torch version 2.6.0).
            # Issue link: https://github.com/pytorch/pytorch/issues/146814.
            # The proper way should be
            # module, name = f.__module__, f.__qualname__
            # rather than the current implementation
            # module, name = f.__module__, f.__name__
            # unregistered_type = (
            #     field_type,
            #     f"{field_type.__module__}.{field_type.__qualname__}",
            # )
            # safe_callables.add(unregistered_type)

            # Once the _get_user_allowed_globals is fixed,
            # the following line should be used instead
            safe_callables.add(field_type)

    # Add all classes to the safe type set
    safe_callables.update(all_classes)

    return safe_callables


def add_sub_class(base_class: type, package_name: str) -> SafeCallables:
    """
    Register all subclasses of base_class found in package as safe callables

    Arguments
    ---------
    base_class: Type
        The base class to find subclasses of
    package_name: str
        The package name to search in (e.g., "ss")
    """

    # Import all submodules to ensure all classes are loaded
    Package.import_submodules(package_name)

    # Get all subclasses
    subclasses = Package.get_subclasses(base_class)

    # Include base class
    all_classes = subclasses.union({base_class})

    # Add all classes and their fields as safe globals
    logger.debug("")
    logger.debug(
        "Add the following classes and their fields "
        "as safe globals for torch.load method:"
    )
    safe_callables = SafeCallables()

    # Add all classes to the safe type set
    safe_callables.update(all_classes)

    return safe_callables


def add_type_var(bound_class: type, package_name: str) -> SafeCallables:
    """Build SafeCallables from a configuration mapping."""
    type_var_config = {
        "TC": "ss.utility.learning.parameter.transformer.config._config",
        "ExpTC": "ss.utility.learning.parameter.transformer.exp.config._config",  # noqa: E501
        "SoftmaxTC": "ss.utility.learning.parameter.transformer.softmax.config._config",  # noqa: E501
        "LinearSoftmaxTC": "ss.utility.learning.parameter.transformer.softmax.linear.config._config",  # noqa: E501
    }

    safe_type_vars: set[SafeCallable] = set()
    for class_name, module_path in type_var_config.items():
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            safe_type_vars.add((cls, f"{module_path}.{class_name}"))
        except (ImportError, AttributeError):
            # Log or handle missing classes
            pass

    return SafeCallables(safe_type_vars)


def add_builtin() -> SafeCallables:
    import operator
    from collections import defaultdict

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
