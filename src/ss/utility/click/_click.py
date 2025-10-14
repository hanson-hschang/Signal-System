import inspect
import json
import re
import types
from collections.abc import Callable
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import click

from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)

FC = TypeVar("FC", bound=Union[Callable[..., Any], click.core.Command])
CC = TypeVar("CC", bound="BaseClickConfig")


@dataclass
class BaseClickConfig:
    def __str__(self) -> str:
        return json.dumps(self.__dict__, indent=len(logger.indent()))

    @classmethod
    def load(cls: type[CC], filepath: Path | None = None, **kwargs: Any) -> CC:
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
                if isinstance(original_value := getattr(config, key), StrEnum):
                    value = type(original_value)[str(value).upper()]
                setattr(config, key, value)
        return config

    @classmethod
    def options(
        cls: type[CC], allow_file_overwrite: bool = False
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
    CustomClickConfig: type[BaseClickConfig], field_name: str
) -> str:
    # Get the source code of the class
    source = inspect.getsource(CustomClickConfig)

    # Look for the field definition and extract any comment
    # Regex components for matching a dataclass field with a comment:
    # - field_name: the name of the field
    # - type_annotation: matches type annotations like 'str', 'list[int]', ...
    # - default_value: matches an optional default value assignment
    # - comment: matches the comment after '#'
    type_annotation = r"(?:\w+|\w+\[.*?\]|[\w.]+)"
    default_value = r"(?:=\s*[^#]*)?"
    comment = r"#\s*(.*)"
    pattern = (
        rf"{field_name}\s*:\s*{type_annotation}\s*{default_value}\s*{comment}"  # noqa: E501
    )
    found = re.search(pattern, source)

    if found and found.group(1):
        return found.group(1).strip()
    else:
        return ""


def extract_choices_from_comment(help_text: str) -> tuple[list | None, str]:
    """Extract choices from help text if it's in format [choice1|choice2]."""

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


def remove_linting(help_text: str) -> str:
    """Remove any linting or formatting artifacts from help text."""
    # Remove noqa comments
    help_text = re.sub(r"#\s*noqa.*", "", help_text).strip()
    # Remove trailing whitespace
    help_text = help_text.rstrip()
    # Remove multiple consecutive spaces
    help_text = re.sub(r"\s{2,}", " ", help_text).strip()
    return help_text


def create_option(
    field_name: str, field_type: Any, help_text: str
) -> Callable[[FC], FC]:
    """Create a Click option from a dataclass field."""
    help_text = remove_linting(help_text)

    choices, help_text = extract_choices_from_comment(help_text)

    # Handle Optional types
    origin_type = get_origin(field_type)
    if origin_type is Union or origin_type is types.UnionType:
        args = get_args(field_type)
        not_none_args = [arg for arg in args if arg is not type(None)]
        if not_none_args:
            field_type = not_none_args[0]
        else:
            field_type = args[0]

    if field_type is bool:
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
