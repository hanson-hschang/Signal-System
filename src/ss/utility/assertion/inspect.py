from typing import Any, Callable, Dict, List, Tuple

import inspect

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
