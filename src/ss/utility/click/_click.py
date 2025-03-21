from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_type_hints,
)

import inspect
import json
import re
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path

import click

from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)

_AnyCallable = Callable[..., Any]
FC = TypeVar("FC", bound=Union[_AnyCallable, click.core.Command])
CC = TypeVar("CC", bound="BaseClickConfig")


@dataclass
class BaseClickConfig:

    def __str__(self) -> str:
        return json.dumps(self.__dict__, indent=len(logger.indent()))

    @classmethod
    def load(
        cls: Type[CC], filepath: Optional[Path] = None, **kwargs: Any
    ) -> CC:
        config = cls()
        if filepath is not None:
            try:
                with open(filepath) as file:
                    config = cls(**json.load(file))
            except (FileNotFoundError, json.JSONDecodeError) as e:
                raise ValueError(
                    f"Failed to load config file from path {filepath}."
                ) from e

        for key, value in kwargs.items():
            if value is not None:
                if isinstance(_value := getattr(config, key), StrEnum):
                    value = type(_value)[str(value).upper()]
                setattr(config, key, value)
        return config

    @classmethod
    def options(
        cls: Type[CC], allow_file_overwrite: bool = False
    ) -> Callable[[FC], FC]:

        def decorator(func: FC) -> FC:

            # Add the config-filepath option
            if allow_file_overwrite:
                func = click.option(
                    "--config-filepath",
                    type=click.Path(),
                    default=None,
                    help="The path to the configuration file.",
                )(func)

            # Add options for each field in the dataclass
            for _field in reversed(fields(cls)):
                # Skip internal or private fields
                if _field.name.startswith("_"):
                    continue
                # Create click option
                field_type = get_type_hints(cls).get(_field.name, str)
                help_text = get_help_text(cls, _field.name)
                field_name = _field.name.replace("_", "-")
                option = create_option(field_name, field_type, help_text)

                # Add the option to the function
                func = option(func)

            return func

        return decorator


def get_help_text(
    CustomClickConfig: Type[BaseClickConfig], field_name: str
) -> str:
    # Get the source code of the class
    source = inspect.getsource(CustomClickConfig)

    # Look for the field definition and extract any comment
    pattern = rf"{field_name}\s*:\s*(?:\w+|\w+\[.*?\]|[\w.]+)\s*(?:=\s*[^#]*)?\s*#\s*(.*)"
    found = re.search(pattern, source)

    if found and found.group(1):
        return found.group(1).strip()
    else:
        return ""


def extract_choices_from_comment(help_text: str) -> Tuple[Optional[List], str]:
    """Extract choices from help text if it's in format [choice1|choice2|choice3]."""

    found = re.search(r"\[\s*([\w\|\s]+)\s*\]", help_text)
    if found and found.group(1):
        # Split the choices by |
        choices = [choice.strip() for choice in found.group(1).split("|")]
        # Remove the choices part from the help text
        cleaned_help_text = re.sub(
            r"\s*\[\s*[\w\|\s]+\s*\]", "", help_text
        ).strip()
        return choices, cleaned_help_text
    return None, help_text


def create_option(
    field_name: str, field_type: Type, help_text: str
) -> Callable[[FC], FC]:
    """Create a Click option from a dataclass field."""
    choices, help_text = extract_choices_from_comment(help_text)

    if field_type == bool:
        option = click.option(f"--{field_name}", is_flag=True, help=help_text)
    elif choices:
        option = click.option(
            f"--{field_name}",
            type=click.Choice(choices, case_sensitive=False),
            help=help_text,
        )
    else:
        option = click.option(
            f"--{field_name}", type=field_type, help=help_text
        )
    return option
