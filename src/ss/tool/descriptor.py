from typing import Generic, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray


class TensorDescriptor:
    def __init__(self, *name_of_dimensions: str) -> None:
        self._name_of_dimensions = list(name_of_dimensions)

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.private_name = "_" + name

    def __get__(self, obj: object, obj_type: type) -> NDArray[np.float64]:
        value: NDArray[np.float64] = getattr(obj, self.private_name)
        return value.copy()

    def __set__(self, obj: object, value: ArrayLike) -> None:
        value = np.array(value, dtype=np.float64)
        shape = tuple(getattr(obj, name) for name in self._name_of_dimensions)
        assert (
            value.shape == shape
        ), f"input shape {value.shape} does not match with {self.name} shape {shape}."
        setattr(obj, self.private_name, value)


class MultiSystemTensorDescriptor:
    def __init__(self, *name_of_dimensions: str) -> None:
        self._name_of_dimensions = list(name_of_dimensions)
        assert (
            len(self._name_of_dimensions) > 1
        ), "At least two dimensions are required."
        assert (
            self._name_of_dimensions[0] == "_number_of_systems"
        ), "The first dimension must be '_number_of_systems'."

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.private_name = "_" + name

    def __get__(self, obj: object, obj_type: type) -> NDArray[np.float64]:
        value: NDArray[np.float64] = getattr(obj, self.private_name)
        if getattr(obj, "_number_of_systems") == 1:
            value = value[0]
        return value.copy()

    def __set__(self, obj: object, value: ArrayLike) -> None:
        value = np.array(value, dtype=np.float64)
        if getattr(obj, "_number_of_systems") == 1:
            value = value[np.newaxis, :]
        shape = tuple(getattr(obj, name) for name in self._name_of_dimensions)
        assert (
            value.shape == shape
        ), f"input shape {value.shape} does not match with {self.name} shape {shape}."
        setattr(obj, self.private_name, value)


T = TypeVar("T")


class ReadOnlyDescriptor(Generic[T]):
    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.private_name = "_" + name

    def __get__(self, obj: object, obj_type: type) -> T:
        value: T = getattr(obj, self.private_name)
        return value
