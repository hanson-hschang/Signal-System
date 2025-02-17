from typing import Any, Dict, Type, TypeVar, cast

T = TypeVar("T", bound="SingletonMeta")


class SingletonMeta(type):
    _instances: Dict[Type, Any] = {}

    def __call__(cls: Type[T], *args: Any, **kwargs: Any) -> T:  # type: ignore
        if cls not in SingletonMeta._instances:
            SingletonMeta._instances[cls] = super().__call__(*args, **kwargs)
        return cast(T, SingletonMeta._instances[cls])
