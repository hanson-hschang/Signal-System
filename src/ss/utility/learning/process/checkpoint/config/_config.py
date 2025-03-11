from typing import Callable, cast

from dataclasses import dataclass, field
from datetime import datetime
from enum import Flag, auto
from pathlib import Path

from ss.utility.assertion.validator import NonnegativeIntegerValidator
from ss.utility.condition import Condition
from ss.utility.descriptor import DataclassDescriptor
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


@dataclass
class CheckpointConfig:

    @dataclass
    class Initial:

        class IndexDescriptor(DataclassDescriptor[int]):
            def __set__(
                self,
                obj: object,
                value: int,
            ) -> None:
                value = NonnegativeIntegerValidator(value).get_value()
                super().__set__(obj, value)

        # skip: bool = False
        index: IndexDescriptor = IndexDescriptor(0)

    @dataclass
    class Appendix:

        class Option(Flag):
            COUNTER = auto()
            DATE = auto()
            TIME = auto()

        option: Option = Option.DATE | Option.TIME

        def __post_init__(self) -> None:
            self._index_digit = 2
            self._index_format = self._create_index_format()
            self._date_format = "_%Y%m%d"
            self._time_format = "_%H%M%S"

        def _create_index_format(self) -> str:
            return "_checkpoint_{:0" + str(self._index_digit) + "d}"

        @property
        def digit(self) -> int:
            return self._index_digit

        @digit.setter
        def digit(self, digit: int) -> None:
            self._index_digit = digit
            self._index_format = self._create_index_format()

        def __call__(self, index: int) -> str:
            now = datetime.now()
            appendix = ""
            if self.Option.COUNTER in self.option:
                if (digit := len(str(index))) > self._index_digit:
                    self.digit = digit
                appendix += self._index_format.format(index)
            if self.Option.DATE in self.option:
                appendix += now.strftime(self._date_format)
            if self.Option.TIME in self.option:
                appendix += now.strftime(self._time_format)
            return appendix

    folderpath: Path = field(default_factory=lambda: Path("checkpoints"))
    filename: Path = field(default_factory=lambda: Path("model"))
    appendix: Appendix = field(default_factory=Appendix)
    initial: Initial = field(
        default_factory=cast(Callable[[], Initial], Initial)
    )
    per_epoch_period: int = 1

    # save_last: bool = True
    # save_best: bool = True

    def __post_init__(self) -> None:
        if not (self.filename.suffix == ""):
            logger.error(
                "The suffix of the checkpoint filename should be empty."
            )
        self._condition = Condition(any)

    def condition(self, epoch: int) -> Condition:
        self._condition(epoch=(epoch % self.per_epoch_period) == 0)
        return self._condition
