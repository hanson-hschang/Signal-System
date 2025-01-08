from typing import Dict, Union

import logging
import sys
from collections import OrderedDict
from enum import IntEnum
from pathlib import Path

from ss.utility.assertion.validator import FilePathValidator
from ss.utility.singleton import SingletonMeta


class Logging(metaclass=SingletonMeta):
    _logger: Dict[str, logging.Logger] = OrderedDict()
    _file_extension = ".log"

    class Level(IntEnum):
        DEBUG = logging.DEBUG
        INFO = logging.INFO
        WARNING = logging.WARNING
        ERROR = logging.ERROR
        CRITICAL = logging.CRITICAL

    @classmethod
    def get_logger(
        cls,
        name: str,
        level: Level = Level.DEBUG,
    ) -> logging.Logger:
        """
        Get a logger with the specified name.

        Arguments:
        ----------
            name: str
                The name of the logger
            level: Level (default: Level.DEBUG)
                The log level of the logger

        Returns:
        --------
            logger: logging.Logger
                The logger with the specified name and log level
        """
        names = name.split(".")
        if (
            (len(names) > 1)
            and (names[-1][0] == "_")
            and (names[-2] in names[-1][1:])
        ):
            name = ".".join(names[:-1])
        if name in cls._logger:
            return cls._logger[name]
        logger = logging.getLogger(name)
        logger.setLevel(level)
        cls._logger[name] = logger
        return logger

    def __init__(
        self,
        filename: Union[str, Path],
        log_level: Level,
        log_format: str,
        datetime_format: str,
        verbose_level: Level,
        verbose_format: str,
    ) -> None:
        # TODO: check and validate the input arguments

        filepath = FilePathValidator(
            filename, self._file_extension
        ).get_filepath()

        # Clear any existing handlers for all loggers
        # and count the maximum length of loggers' names
        max_name_length = 0
        for name, logger in self._logger.items():
            max_name_length = max(max_name_length, len(name))
            logger.handlers = []

        # Create formatters and change the format of loggers' name to have a fixed width
        log_format = log_format.replace(
            "%(name)s", f"%(name){max_name_length}s"
        )
        verbose_format = verbose_format.replace(
            "%(name)s", f"%(name){max_name_length}s"
        )
        file_formatter = logging.Formatter(
            fmt=log_format,
            datefmt=datetime_format,
        )
        console_formatter = logging.Formatter(
            fmt=verbose_format,
        )

        # File handler for all logs
        self.file_handler = logging.FileHandler(filepath)
        self.file_handler.setLevel(log_level)
        self.file_handler.setFormatter(file_formatter)

        # Console handler (stdout) for verbose output
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setLevel(verbose_level)
        self.console_handler.setFormatter(console_formatter)

        logging_logger = self._logger[__name__]
        for name, logger in self._logger.items():
            logger.addHandler(self.console_handler)
            logger.addHandler(self.file_handler)
            logging_logger.debug(f"logger: {name} has been initialized")
        logging_logger.info(
            f"logging has been initialized: verbose_level = {verbose_level.name} and log_level = {log_level.name}"
        )
        logging_logger.info(f"logging file = {str(filename)}")

        # Store levels for property access
        self._verbose_level = verbose_level
        self._log_level = log_level

        # TODO: add descriptor for log_level and verbose_level

    @classmethod
    def basic_config(
        cls,
        filename: Union[str, Path],
        log_level: Level = Level.INFO,
        log_format: str = r"%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datetime_format: str = r"%Y-%m-%d %H:%M:%S",
        verbose_level: Level = Level.WARNING,
        verbose_format: str = "%(name)s | %(levelname)s | %(message)s",
    ) -> "Logging":
        return cls(
            filename,
            log_level,
            log_format,
            datetime_format,
            verbose_level,
            verbose_format,
        )


logger = Logging.get_logger(__name__)
