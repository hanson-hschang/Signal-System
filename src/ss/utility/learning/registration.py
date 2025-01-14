from typing import List, Set, Type

import importlib
import pkgutil

import torch

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


def register_subclasses(base_class: Type, package_name: str) -> List[Type]:
    """
    Register all subclasses of base_class found in the package.

    Arguments
    ---------
    base_class: Type
        The base class to find subclasses of
    package_name: str
        The package name to search in (e.g., "ss")

    Returns
    -------
    all_classes: List[Type]
        List of registered classes
    """

    # Import all submodules to ensure all classes are loaded
    import_submodules(package_name)

    # Get all subclasses
    subclasses = get_subclasses(base_class)

    # Include base class
    all_classes = list(subclasses.union({base_class}))
    logger.debug(
        "Register the following classes as safe globals for torch.load method:"
    )
    for cls in all_classes:
        logger.debug(f"    { cls }")

    # Add all classes to safe globals
    torch.serialization.add_safe_globals(all_classes)

    return all_classes


def register_numpy() -> None:
    # FIXME: This is a temporary fix to allow for safe loading of numpy scalars
    # and dtypes. This should be removed since there is no numpy dependency in
    # the .pt files currently.

    # Add numpy scalar to safe globals before loading
    from numpy import dtype
    from numpy.dtypes import Float64DType, Int64DType

    torch.serialization.add_safe_globals([dtype, Int64DType, Float64DType])
