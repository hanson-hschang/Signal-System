from typing import Generic, TypeVar

T = TypeVar("T")


class ReadOnlyDescriptor(Generic[T]):
    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.private_name = "_" + name

    def __get__(self, obj: object, obj_type: type) -> T:
        value: T = getattr(obj, self.private_name)
        return value


class Descriptor(ReadOnlyDescriptor[T]):
    def __set__(self, obj: object, value: T) -> None:
        setattr(obj, self.private_name, value)
