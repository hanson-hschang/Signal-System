from typing import List, Set, Type

import importlib
import pkgutil

import torch

from ss.utility.assertion.inspect import get_nondefault_type_fields
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


def import_submodules(module_name: str, parent_module_name: str = "") -> None:
    """
    Import all submodules of a module recursively

    Arguments
    ---------
    module_name: str
        The module name to import submodules of
    parent_module_name: str
        The parent module name
    """

    # Import the module
    module_name = (
        f"{parent_module_name}.{module_name}"
        if parent_module_name
        else module_name
    )
    module = importlib.import_module(module_name)

    # Check if module is a package
    if hasattr(module, "__path__"):
        # Iterate through all submodules
        submodule_name_list = []
        for _, submodule_name, _ in pkgutil.walk_packages(module.__path__):
            names = submodule_name.split(".")
            # FIXME: Not sure why some submodule_name are in the format of
            # "some_module_name.module_name" and some are in the format of
            # "module_name". This is a temporary fix to avoid the former.
            if len(names) > 1:
                continue
            submodule_name_list.append(submodule_name)
        for submodule_name in submodule_name_list:
            import_submodules(
                module_name=f"{submodule_name}",
                parent_module_name=module_name,
            )


def get_subclasses(cls: Type) -> Set[Type]:
    """
    Get all subclasses recursively

    Arguments
    ---------
    cls: Type
        The base class to find subclasses of


    Returns
    -------
    subclasses: Set[Type]
        Set of all subclasses
    """
    subclasses = set()
    for subclass in cls.__subclasses__():
        subclasses.add(subclass)
        subclasses.update(get_subclasses(subclass))
    return subclasses


def resolve_module_name(module_name: str) -> str:
    """
    Resolve the module name to the full module path

    Arguments
    ---------
    module_name: str
        The module name to resolve

    Returns
    -------
    full_module_name: str
        The full module path
    """
    # names = module_name.split(".")
    # if len(names) == 1:
    #     return module_name
    # module_name = names[0]
    # for name in names[1:]:
    #     if name[0] == "_":
    #         break
    #     module_name = f"{module_name}.{name}"

    return module_name


def register_subclasses(base_class: Type, package_name: str) -> None:
    """
    Register all subclasses of base_class found in the package.

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

    # Register all classes and their fields as safe globals
    logger.debug(
        "Register the following classes and their fields as safe globals for torch.load method:"
    )
    unregistered_types: Set = set()
    for cls in all_classes:
        logger.debug(
            f"    { cls.__qualname__ } "
            f"( from { resolve_module_name(cls.__module__) } module )"
        )
        unregistered_fields = get_nondefault_type_fields(cls)
        for field_name, field_type in unregistered_fields.items():
            logger.debug(
                f"        { field_name }: "
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
            unregistered_types.add(unregistered_type)
            # Once the _get_user_allowed_globals is fixed, the following line should be used instead
            # unregistered_types.add(field_type)

    torch.serialization.add_safe_globals(list(all_classes))
    torch.serialization.add_safe_globals(list(unregistered_types))


def register_builtin() -> None:
    torch.serialization.add_safe_globals(
        [getattr, setattr]
    )  # This is for getter and setter of properties


def register_numpy() -> None:
    # FIXME: This is a temporary fix to allow for safe loading of numpy scalars
    # and dtypes. This should be removed since there is no numpy dependency in
    # the .pt files currently.

    # Add numpy scalar to safe globals before loading
    from numpy import dtype
    from numpy.dtypes import Float64DType, Int64DType

    torch.serialization.add_safe_globals([dtype, Int64DType, Float64DType])
