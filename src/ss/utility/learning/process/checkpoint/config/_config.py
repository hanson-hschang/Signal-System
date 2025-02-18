from dataclasses import dataclass, field
from datetime import datetime
from enum import Flag, auto
from pathlib import Path

from ss.utility.condition import Condition
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


@dataclass
class CheckpointConfig:

    @dataclass
    class Appendix:

        class Option(Flag):
            COUNTER = auto()
            DATE = auto()
            TIME = auto()

        option: Option = Option.DATE | Option.TIME

        def __post_init__(self) -> None:
            self._counter_digit = 2
            self._counter_format = self._create_counter_format()
            self._date_format = "_%Y%m%d"
            self._time_format = "_%H%M%S"

        def _create_counter_format(self) -> str:
            return "_checkpoint_{:0" + str(self._counter_digit) + "d}"

        @property
        def digit(self) -> int:
            return self._counter_digit

        @digit.setter
        def digit(self, digit: int) -> None:
            self._counter_digit = digit
            self._counter_format = self._create_counter_format()

        def __call__(self, counter: int) -> str:
            now = datetime.now()
            appendix = ""
            if self.Option.COUNTER in self.option:
                if (digit := len(str(counter))) > self._counter_digit:
                    self.digit = digit
                appendix += self._counter_format.format(counter)
            if self.Option.DATE in self.option:
                appendix += now.strftime(self._date_format)
            if self.Option.TIME in self.option:
                appendix += now.strftime(self._time_format)
            return appendix

    folderpath: Path = field(default_factory=lambda: Path("checkpoints"))
    filename: Path = field(default_factory=lambda: Path("model"))
    appendix: Appendix = field(default_factory=Appendix)
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
