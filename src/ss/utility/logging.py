from typing import Dict, Optional, Union

import logging
import sys
from collections import OrderedDict
from enum import IntEnum
from pathlib import Path

from ss.utility.singleton import SingletonMeta


class Logging(metaclass=SingletonMeta):
    _logger: Dict[str, logging.Logger] = OrderedDict()

    class LogLevel(IntEnum):
        DEBUG = logging.DEBUG
        INFO = logging.INFO
        WARNING = logging.WARNING
        ERROR = logging.ERROR
        CRITICAL = logging.CRITICAL

    @classmethod
    def get_logger(
        cls,
        name: str,
        log_level: LogLevel = LogLevel.DEBUG,
    ) -> logging.Logger:
        """
        Get a logger with the specified name.

        Arguments
        ---------
            name: str
                The name of the logger
            log_level: LogLevel (default: LogLevel.DEBUG)
                The log level of the logger

        Returns
        -------
            logger: logging.Logger
                The logger with the specified name and log level
        """
        if name in cls._logger:
            return cls._logger[name]
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        cls._logger[name] = logger
        return logger

    def __init__(
        self,
        filename: Union[str, Path],
        log_level: LogLevel = LogLevel.DEBUG,
        log_format: str = r"%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datetime_format: str = r"%Y-%m-%d %H:%M:%S",
        verbose_level: LogLevel = LogLevel.INFO,
        verbose_format: str = "%(name)s | %(levelname)s | %(message)s",
    ) -> None:
        # TODO: check and validate the input arguments

        # Clear any existing handlers for all loggers
        # and count the maximum length of loggers' names
        max_name_length = 0
        for name, logger in self._logger.items():
            max_name_length = max(max_name_length, len(name))
            logger.handlers = []
        max_name_length += 1

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
        self.file_handler = logging.FileHandler(filename)
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
            logging_logger.debug(f"logger: {name} has been initialized.")

        # Store levels for property access
        self._verbose_level = verbose_level
        self._log_level = log_level

        main_logger = self.get_logger("__main__")
        main_logger.info(
            f"logging has been initialized with the verbose_level = {verbose_level.name}, "
        )
        main_logger.info(
            f"and log_level = {log_level.name} with the filename as {str(filename)}"
        )

        # TODO: add descriptor for log_level and verbose_level


logger = Logging.get_logger(__name__)
