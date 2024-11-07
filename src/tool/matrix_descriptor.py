import numpy as np
from numpy.typing import ArrayLike, NDArray


class MatrixDescriptor:
    def __init__(
        self, name_of_row_numbers: str, name_of_column_numbers: str
    ) -> None:
        self.name_of_row_numbers = name_of_row_numbers
        self.name_of_column_numbers = name_of_column_numbers

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.private_name = "_" + name

    def __get__(self, obj: object, obj_type: type) -> NDArray[np.float64]:
        value: NDArray = getattr(obj, self.private_name)
        if value.shape[0] == 1:
            value = value.squeeze()
        return value

    def __set__(self, obj: object, value: ArrayLike) -> None:
        value = np.array(value, dtype=np.float64)
        if len(value.shape) == 1:
            value = value.reshape((1, -1))
        shape = (
            getattr(obj, self.name_of_row_numbers),
            getattr(obj, self.name_of_column_numbers),
        )
        assert (
            value.shape == shape
        ), f"input shape {value.shape} does not match with {self.name} shape {shape}."
        setattr(obj, self.private_name, value)
