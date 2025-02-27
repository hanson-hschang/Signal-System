from typing import Any, Generic, TypeVar

T = TypeVar("T")
O = TypeVar("O", bound=object)


class ConditionReadOnlyDescriptor(Generic[T, O]):
    def __set_name__(self, obj_type: type, name: str) -> None:
        self.name = name
        self.private_name = "_" + name

    def __get__(self, obj: O, obj_type: type) -> T:
        value: T = getattr(obj, self.private_name)
        return value


class ConditionDescriptor(ConditionReadOnlyDescriptor[T, O]):
    def __set__(self, obj: O, value: T) -> None:
        setattr(obj, self.private_name, value)


class ReadOnlyDescriptor(ConditionReadOnlyDescriptor[T, Any]): ...


class Descriptor(ConditionDescriptor[T, Any]): ...
