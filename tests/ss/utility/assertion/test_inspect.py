import numpy as np
import pytest
from numpy.typing import NDArray

from ss.utility.assertion.inspect import inspect_arguments


def test_inspect_arguments() -> None:

    def func(
        a: NDArray[np.float64], b: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return a + b

    arg_name_shape_dict = {"a": (1,), "b": (1,)}
    result_func = inspect_arguments(
        func=func,
        result_shape=(1,),
        arg_name_shape_dict=arg_name_shape_dict,
    )
    assert result_func(np.array([1]), np.array([2])) == np.array([3])

    with pytest.raises(AssertionError):
        inspect_arguments(
            func=func,
            result_shape=(1,),
            arg_name_shape_dict={"a": (1,)},
        )
