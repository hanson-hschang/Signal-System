from dataclasses import dataclass

import psutil

from ss.utility.device.monitor.performance import Performance


@dataclass
class Memory:
    total: int
    available: int
    percent: float
    used: int
    free: int
    active: int
    inactive: int
    wired: int


class CpuPerformance(Performance):
    def __init__(self) -> None:
        self._cpu_process = psutil.Process()
        super().__init__()

    def _process(self) -> None:
        self._utilization = self._get_utilization()
        self._memory = self._get_memory()

    @property
    def utilization(self) -> float:
        return self._utilization

    @property
    def temperature(self) -> float:
        return float("nan")

    @property
    def total_memory(self) -> int:
        return self._memory.total

    @property
    def free_memory(self) -> int:
        return self._memory.free

    @property
    def used_memory(self) -> int:
        return self._memory.used

    @property
    def memory_percent(self) -> float:
        return self._memory.percent

    def _get_utilization(self) -> float:
        utilization: float = self._cpu_process.cpu_percent()
        return utilization

    def _get_memory(self) -> Memory:
        memory = psutil.virtual_memory()
        return Memory(
            total=memory.total,
            available=memory.available,
            percent=memory.percent,
            used=memory.used,
            free=memory.free,
            active=memory.active,
            inactive=memory.inactive,
            wired=memory.wired,
        )
