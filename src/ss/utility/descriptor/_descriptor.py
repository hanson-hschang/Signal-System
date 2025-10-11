from typing import Generic, TypeVar

T = TypeVar("T")
O = TypeVar("O", bound=object, default=object)  # noqa: E741


class ReadOnlyDescriptor(Generic[T, O]):
    def __set_name__(self, owner: type[O], name: str) -> None:
        self.name = name
        self.private_name = "_" + name

    def _validate_instance(self, instance: O) -> None:
        pass

    def __get__(self, instance: O | None, owner: type[O]) -> T:
        if instance is None:
            raise AttributeError(f"'{self.name}' is not set.")
        self._validate_instance(instance)
        value: T = getattr(instance, self.private_name)
        return value


class Descriptor(ReadOnlyDescriptor[T, O], Generic[T, O]):
    def __set__(self, instance: O, value: T) -> None:
        self._validate_instance(instance)
        setattr(instance, self.private_name, value)


class ReadOnlyDataclassDescriptor(Generic[T, O]):
    def __init__(self, value: T | None = None) -> None:
        self._default_value = value

    def __set_name__(self, owner: type[O], name: str) -> None:
        self.name = name
        self.private_name = "_" + name

    def _validate_instance(self, instance: O) -> None:
        pass

    def __get__(self, instance: O | None, owner: type[O]) -> T:
        if instance is None:
            if self._default_value is None:
                raise AttributeError(f"'{self.name}' is not set.")
            return self._default_value
        self._validate_instance(instance)
        if not hasattr(instance, self.private_name):
            setattr(instance, self.private_name, self._default_value)
        value: T = getattr(instance, self.private_name)
        return value


class DataclassDescriptor(ReadOnlyDataclassDescriptor[T, O], Generic[T, O]):
    def __set__(self, instance: O, value: T) -> None:
        self._validate_instance(instance)
        setattr(instance, self.private_name, value)
