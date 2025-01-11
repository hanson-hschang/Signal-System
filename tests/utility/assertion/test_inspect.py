import numpy as np
import pytest
from numpy.typing import NDArray

from ss.utility.assertion.inspect import inspect_arguments


def test_inspect_arguments() -> None:
    def func(a: NDArray, b: NDArray) -> NDArray:
        result: NDArray = a + b
        return result

    arg_name_shape_dict = {"a": (1,), "b": (1,)}
    result_func = inspect_arguments(
        func=func,
        arg_name_shape_dict=arg_name_shape_dict,
        result_shape=(1,),
    )
    assert result_func(np.array([1]), np.array([2])) == np.array([3])

    with pytest.raises(AssertionError):
        inspect_arguments(
            func=func,
            arg_name_shape_dict={"a": (1,)},
            result_shape=(1,),
        )
