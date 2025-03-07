from typing import Generic, TypeVar

from ss.utility.callback import Callback
from ss.utility.data import MetaInfo
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class Performance:
    def __init__(self, device_name: str = "device") -> None:
        self._device_name = device_name
        self._process()

    @property
    def device_name(self) -> str:
        return self._device_name

    def _process(self) -> None:
        raise NotImplementedError

    @property
    def utilization(self) -> float:
        raise NotImplementedError

    @property
    def temperature(self) -> float:
        raise NotImplementedError

    @property
    def total_memory(self) -> int:
        raise NotImplementedError

    @property
    def free_memory(self) -> int:
        raise NotImplementedError

    @property
    def used_memory(self) -> int:
        raise NotImplementedError

    @property
    def memory_percent(self) -> float:
        raise NotImplementedError

    def process(self, current_step: int) -> int:
        self._process()
        return current_step + 1


P = TypeVar("P", bound=Performance)


class PerformanceCallback(Callback, Generic[P]):
    def __init__(self, performance: P) -> None:
        super().__init__(step_skip=1)
        self._performance = performance
        self._device_name = self._performance.device_name
        self.add_meta_info(
            MetaInfo(
                **{
                    self._device_name
                    + "/total_memory": self._performance.total_memory
                },
            )
        )

    def _record(self, time: float) -> None:
        super()._record(time)
        self._callback_params[self._device_name + "/utilization"].append(
            self._performance.utilization
        )
        self._callback_params[self._device_name + "/temperature"].append(
            self._performance.temperature
        )
        self._callback_params[self._device_name + "/free_memory"].append(
            self._performance.free_memory
        )
        self._callback_params[self._device_name + "/used_memory"].append(
            self._performance.used_memory
        )
        self._callback_params[self._device_name + "/memory_percent"].append(
            self._performance.memory_percent
        )
