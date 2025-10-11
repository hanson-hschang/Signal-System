import logging
import sys
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from types import TracebackType
from typing import Optional, TypeAlias, Union, override

from tqdm import tqdm

from ss.utility.assertion.validator import FilePathValidator
from ss.utility.singleton import SingletonMeta

ExcInfoType: TypeAlias = Union[
    bool,
    tuple[
        type[BaseException],
        BaseException,
        Optional[TracebackType],
    ],
    tuple[None, None, None],
    BaseException,
]


class LogLevel(IntEnum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogfileSetting:
    filename: str | Path
    log_level: LogLevel
    log_format: str
    datetime_format: str
    _file_extension: str = field(default=".log", init=False)

    def __post_init__(self) -> None:
        self.filepath = FilePathValidator(
            self.filename, self._file_extension
        ).get_filepath()


@dataclass
class VerboseSetting:
    log_level: LogLevel
    log_format: str


class Logger(logging.Logger):
    def __init__(self, name: str, level: LogLevel) -> None:
        super().__init__(name, level)

    def progress_bar(
        self,
        iterable: Iterable,
        *,
        total: int | None = None,
        show_progress: bool = True,
    ) -> tqdm:
        return tqdm(iterable, total=total, disable=not show_progress)

    # This is a temporary solution to indent the log messages
    def indent(self, level: int = 1) -> str:
        return "    " * level

    @override
    def debug(
        self,
        message: object,
        *args: object,
        indent_level: int = 0,
        exc_info: ExcInfoType | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        messages = str(message).split("\n")
        for _message in messages:
            super().debug(
                self.indent(indent_level) + _message,
                *args,
                exc_info=exc_info,
                stack_info=stack_info,
                stacklevel=stacklevel,
                extra=extra,
            )

    @override
    def info(
        self,
        message: object,
        *args: object,
        indent_level: int = 0,
        exc_info: ExcInfoType | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        messages = str(message).split("\n")
        for _message in messages:
            super().info(
                self.indent(indent_level) + _message,
                *args,
                exc_info=exc_info,
                stack_info=stack_info,
                stacklevel=stacklevel,
                extra=extra,
            )

    def error(
        self,
        message: object,
        *args: object,
        indent_level: int = 0,
        exc_info: ExcInfoType | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        messages = str(message).split("\n")
        for _message in messages:
            super().error(
                self.indent(indent_level) + _message,
                *args,
                exc_info=exc_info,
                stack_info=stack_info,
                stacklevel=stacklevel,
                extra=extra,
            )
        raise ValueError(message)


class Logging(metaclass=SingletonMeta):
    _logger: dict[str, Logger] = OrderedDict()

    @classmethod
    def get_logger(
        cls,
        name: str,
        level: LogLevel = LogLevel.DEBUG,
    ) -> Logger:
        """
        Get a logger with the specified name.

        Arguments:
        ----------
            name: str
                The name of the logger
            level: LogLevel (default: LogLevel.DEBUG)
                The log level of the logger

        Returns:
        --------
            logger: Logger
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
        logger = Logger(name, level)
        cls._logger[name] = logger
        return logger

    def __init__(
        self,
        logfile_setting: LogfileSetting,
        verbose_setting: VerboseSetting,
    ) -> None:
        # TODO: check and validate the input arguments

        self.logfile_setting = logfile_setting
        self.verbose_setting = verbose_setting

        # Clear any existing handlers for all loggers
        # and count the maximum length of loggers' names
        max_name_length = 0
        for name, logger in self._logger.items():
            max_name_length = max(max_name_length, len(name))
            logger.handlers = []

        # Create formatters with a fixed width for loggers' names
        self.logfile_setting.log_format = (
            self.logfile_setting.log_format.replace(
                "%(name)s", f"%(name){max_name_length}s"
            )
        )
        self.verbose_setting.log_format = (
            self.verbose_setting.log_format.replace(
                "%(name)s", f"%(name){max_name_length}s"
            )
        )
        file_formatter = logging.Formatter(
            fmt=self.logfile_setting.log_format,
            datefmt=self.logfile_setting.datetime_format,
        )
        console_formatter = logging.Formatter(
            fmt=self.verbose_setting.log_format,
        )

        # File handler for all logs
        self.file_handler = logging.FileHandler(self.logfile_setting.filepath)
        self.file_handler.setLevel(self.logfile_setting.log_level)
        self.file_handler.setFormatter(file_formatter)

        # Console handler (stdout) for verbose output
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setLevel(self.verbose_setting.log_level)
        self.console_handler.setFormatter(console_formatter)

        logging_logger = self._logger[__name__]
        for name, logger in self._logger.items():
            logger.addHandler(self.console_handler)
            logger.addHandler(self.file_handler)
            logging_logger.debug(f"logger: {name} has been initialized")
        logging_logger.info(
            "logging has been initialized with the following settings:"
        )
        logging_logger.info(
            f"    verbose level = {self.verbose_setting.log_level.name}"
        )
        logging_logger.info(
            f"    logfile level = {self.logfile_setting.log_level.name}"
        )
        logging_logger.info(f"logfile path = {self.logfile_setting.filepath}")

    @classmethod
    def basic_config(
        cls,
        filename: str | Path,
        verbose: bool = False,
        debug: bool = False,
        verbose_format: str | None = None,
        logfile_format: str | None = None,
        datetime_format: str | None = None,
    ) -> "Logging":
        if verbose_format is None:
            verbose_format = r"%(name)s | %(levelname)s | %(message)s"
        if logfile_format is None:
            logfile_format = (
                r"%(asctime)s | %(name)s | %(levelname)-8s | %(message)s"
            )
        if datetime_format is None:
            datetime_format = r"%Y-%m-%d %H:%M:%S"
        return cls(
            LogfileSetting(
                filename=filename,
                log_level=LogLevel.DEBUG if debug else LogLevel.INFO,
                log_format=logfile_format,
                datetime_format=datetime_format,
            ),
            VerboseSetting(
                log_level=LogLevel.INFO if verbose else LogLevel.WARNING,
                log_format=verbose_format,
            ),
        )


logger = Logging.get_logger(__name__)
