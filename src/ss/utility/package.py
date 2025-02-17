from typing import Set, Type

import importlib
import pkgutil

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
