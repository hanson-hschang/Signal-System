from typing import Any, Dict, Type, TypeVar

from dataclasses import dataclass

from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)

BLC = TypeVar("BLC", bound="BaseLearningConfig")


@dataclass
class BaseLearningConfig:

    @classmethod
    def reload(
        cls: Type[BLC], config: BLC, name: str = "config", level: int = 0
    ) -> BLC:
        """
        Reload the configuration to ensure that the configuration is updated.
        """
        # Do not use asdict(self) method for conversion because it does not work
        # for different versions due to inconsistent arguments. Instead, use the
        # meta method self.__dict__.

        # return self.__class__(**self.__dict__)

        if level == 0:
            logger.debug("")
            logger.debug("Loading configuration...")
        config_init_arguments: Dict[str, Any] = {}
        config_private_attributes: Dict[str, Any] = {}
        indent_level = level + 1
        logger.debug(logger.indent(indent_level) + f"{name} = " + cls.__name__)
        for key, value in config.__dict__.items():
            # The following condition did not check the dunder attributes
            is_init_argument = True
            if key.startswith("_"):
                key = key[1:]
                is_init_argument = False

            if isinstance(value, BaseLearningConfig):
                value = type(value).reload(value, name=key, level=level + 1)

            if is_init_argument:
                # Did not check the condition that the key exists
                # in the old version but not in the new version
                config_init_arguments[key] = value
            else:
                config_private_attributes[key] = value

            if not isinstance(value, BaseLearningConfig):
                logger.debug(
                    logger.indent(indent_level + 1) + key + " = " + str(value)
                )

        # Create a new instance of the configuration class
        config = cls(**config_init_arguments)

        # Set private attribute through the property setter
        for key, value in config_private_attributes.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config
