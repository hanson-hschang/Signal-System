import torch
from numpy.typing import ArrayLike


class TensorReadOnlyDescriptor:
    def __init__(self, *name_of_dimensions: str) -> None:
        self._name_of_dimensions = list(name_of_dimensions)

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.private_name = "_" + name

    def __get__(self, obj: object, obj_type: type) -> torch.Tensor:
        value: torch.Tensor = getattr(obj, self.private_name)
        return value.detach()


class TensorDescriptor(TensorReadOnlyDescriptor):

    def __set__(self, obj: object, value: ArrayLike) -> None:
        value = torch.tensor(value, dtype=torch.float64)
        shape = tuple(getattr(obj, name) for name in self._name_of_dimensions)
        assert value.shape == shape, (
            f"{self.name} must be in the shape of {shape}. "
            f"{self.name} given has the shape of {value.shape}."
        )
        setattr(obj, self.private_name, value)
