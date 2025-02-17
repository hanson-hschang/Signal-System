from typing import TypeVar

from dataclasses import dataclass

BLC = TypeVar("BLC", bound="BaseLearningConfig")


@dataclass
class BaseLearningConfig:

    def reload(self: BLC) -> BLC:
        """
        Reload the configuration to ensure that the configuration is updated.
        """
        # Do not use asdict(self) method for conversion because it does not work
        # for different versions due to inconsistent arguments. Instead, use the
        # meta method self.__dict__.
        return self.__class__(**self.__dict__)
