from typing import Any, List, Set, Type

import importlib
import inspect
import pkgutil

import torch

from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


def import_submodules(module_name: str) -> None:
    """
    Import all submodules of a module recursively

    Arguments
    ---------
        module_name: str
            The module name to import submodules of
    """
    # FIXME: The following test module check should not be necessary
    # but is required to avoid an error when running tests.
    # Check if module is a test module
    names = module_name.split(".")
    if "test" == names[-1][:4]:
        return
    # FIXME: The following __main__ file check should not be necessary
    # but is required to avoid an error when running.
    # Check if module is a __main__ file
    if "__main__" == names[-1]:
        return

    # Import the module
    module = importlib.import_module(module_name)

    # Check if module is a package
    if hasattr(module, "__path__"):
        # Iterate through all submodules
        for _, name, _ in pkgutil.walk_packages(module.__path__):
            import_submodules(module_name=f"{module_name}.{name}")


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
        List of registered classes
    """

    # Import all submodules to ensure all classes are loaded
    import_submodules(package_name)

    # Get all subclasses
    subclasses = get_subclasses(base_class)

    # Include base class
    all_classes = list(subclasses.union({base_class}))

    # Add all classes to safe globals
    torch.serialization.add_safe_globals(all_classes)

    return all_classes


def register_numpy() -> None:
    # FIXME: This is a temporary fix to allow for safe loading of numpy scalars
    # and dtypes. This should be removed since there is no numpy dependency in
    # the .pt files currently.

    # Add numpy scalar to safe globals before loading
    from numpy import dtype

    # from numpy.core.multiarray import scalar  # type: ignore
    from numpy.dtypes import Float64DType, Int64DType

    torch.serialization.add_safe_globals([dtype, Int64DType])
