from collections import OrderedDict
from collections.abc import Callable, Iterable
from typing import Self


class Condition:
    def __init__(self, quantifier: Callable[[Iterable[bool]], bool]) -> None:
        self.quantifier = quantifier
        self._condition: OrderedDict[str, bool] = OrderedDict()

    def __call__(self, **kwargs: bool) -> Self:
        self._condition.update(kwargs)
        return self

    def reset(self) -> None:
        self._condition.clear()

    def satisfied(self) -> bool:
        return self.quantifier(self._condition.values())

    def not_satisfied(self) -> bool:
        return not self.satisfied()
