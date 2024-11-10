from typing import Generic, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray


class TensorDescriptor:
    def __init__(self, *name_of_dimensions: str) -> None:
        self._name_of_dimensions = list(name_of_dimensions)
        self._get_tensor = self._get_tensor_init
        self._set_tensor = self._set_tensor_init

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.private_name = "_" + name

    def __get__(self, obj: object, obj_type: type) -> NDArray[np.float64]:
        value: NDArray[np.float64] = self._get_tensor(obj, obj_type)
        return value

    def __set__(self, obj: object, value: ArrayLike) -> None:
        self._set_tensor(obj, value)

    def _get_shape(self, obj: object) -> tuple:
        shape = tuple(getattr(obj, name) for name in self._name_of_dimensions)
        return shape

    def _squeeze_first_dimension(self, obj: object) -> None:
        shape = self._get_shape(obj)
        if (self._name_of_dimensions[0] == "_number_of_systems") and (
            shape[0] == 1
        ):
            self._get_tensor = self._get_tensor_squeeze_first_dimension
            self._set_tensor = self._set_tensor_squeeze_first_dimension
            self._name_of_dimensions = self._name_of_dimensions[1:]
        else:
            self._get_tensor = self._get_tensor_complete_shape
            self._set_tensor = self._set_tensor_complete_shape

    def _get_tensor_init(
        self, obj: object, obj_type: type
    ) -> NDArray[np.float64]:
        self._squeeze_first_dimension(obj)
        value: NDArray[np.float64] = self._get_tensor(obj, obj_type)
        return value

    def _get_tensor_squeeze_first_dimension(
        self, obj: object, obj_type: type
    ) -> NDArray[np.float64]:
        value: NDArray[np.float64] = getattr(obj, self.private_name)
        value = value[0]
        return value

    def _get_tensor_complete_shape(
        self, obj: object, obj_type: type
    ) -> NDArray[np.float64]:
        value: NDArray[np.float64] = getattr(obj, self.private_name)
        return value

    def _set_tensor_init(self, obj: object, value: ArrayLike) -> None:
        self._squeeze_first_dimension(obj)
        self._set_tensor(obj, value)

    def _set_tensor_squeeze_first_dimension(
        self, obj: object, value: ArrayLike
    ) -> None:
        value = np.array(value, dtype=np.float64)
        shape = self._get_shape(obj)
        assert (
            value.shape == shape
        ), f"input shape {value.shape} does not match with {self.name} shape {shape}."
        value = value[np.newaxis, :]
        setattr(obj, self.private_name, value)

    def _set_tensor_complete_shape(self, obj: object, value: ArrayLike) -> None:
        value = np.array(value, dtype=np.float64)
        shape = self._get_shape(obj)
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
