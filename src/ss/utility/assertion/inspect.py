from typing import Any, Callable, Dict, List, Set, Tuple, get_type_hints

import inspect
from dataclasses import dataclass, is_dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

_array_like_types: tuple = (
    ArrayLike,
    NDArray,
    NDArray,
    List,
    Tuple,
    np.ndarray,
    list,
    tuple,
    np.ndarray[Any, np.dtype[np.float64]],
)


# Default Python types including common collections
_default_python_types: Set = {
    str,
    int,
    float,
    bool,
    bytes,
    list,
    dict,
    set,
    tuple,
    List,
    Set,
    Dict,
    Tuple,
    Tuple[int, ...],
}


def inspect_arguments(
    func: Callable,
    arg_name_shape_dict: Dict[str, Tuple],
    result_shape: Tuple,
) -> Callable:
    signature = inspect.signature(func)
    arg_dict = {}
    for arg_name in arg_name_shape_dict.keys():
        assert (
            arg_name in signature.parameters
        ), f"{arg_name} should be an argument for {func.__name__}"
        param = signature.parameters[arg_name]
        assert param.annotation in _array_like_types, (
            f"{arg_name} should be of type ArrayLike "
            f"but is of type {param.annotation}"
        )
        arg_dict[arg_name] = np.zeros(arg_name_shape_dict[arg_name])
    try:
        result: NDArray = func(**arg_dict)
    except TypeError as e:
        raise AssertionError(e)
    assert (
        result.shape == result_shape
    ), f"Function {func.__name__} does not return an array of shape {result_shape}"
    return func


def get_nondefault_type_fields(
    cls: type,
) -> Dict[str, type]:
    """
    Returns a dictionary of non-default types found in the dataclass's fields.
    The keys are the field names and the values are the non-default types.

    Parameters
    ----------
    cls : type
        The dataclass to analyze.

    Returns
    -------
    Dict[str, type]
        A dictionary of non-default types found in the dataclass's fields.

    Raises
    ------
    ValueError
        If the input is not a dataclass
    """
    nondefault_type_parameters: Dict[str, type] = {}
    if not is_dataclass(cls):
        raise ValueError("Argument cls must be a dataclass")

    type_hints: Dict[str, type] = get_type_hints(cls)

    for field_name, field_type in type_hints.items():
        if field_type not in _default_python_types:
            nondefault_type_parameters[field_name] = field_type

    return nondefault_type_parameters
