from typing import List, Set, Type

import importlib
import pkgutil

import torch


def import_submodules(package_name: str) -> None:
    """
    Import all submodules of a package recursively

    Arguments
    ---------
        package_name: str
            The package name to import submodules of
    """
    package = importlib.import_module(package_name)

    if hasattr(package, "__path__"):
        for _, name, is_pkg in pkgutil.walk_packages(package.__path__):
            full_name = f"{package_name}.{name}"
            importlib.import_module(full_name)
            if is_pkg:
                import_submodules(full_name)


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
            The package name to search in (e.g., "lss")

    Returns
    -------
        List of registered classes
    """
    # First import all submodules to ensure all classes are loaded
    import_submodules(package_name)

    # Get all subclasses
    subclasses = get_subclasses(base_class)

    # Include base class
    all_classes = list(subclasses.union({base_class}))

    # Add all classes to safe globals
    torch.serialization.add_safe_globals(all_classes)

    return all_classes
