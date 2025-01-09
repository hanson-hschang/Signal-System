from typing import Any


class SingletonMeta(type):
    _instances: dict = {}

    def __call__(cls, *args, **kwargs) -> Any:  # type: ignore
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
