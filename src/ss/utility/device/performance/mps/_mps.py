from ss.utility.device.performance import Performance


class MpsPerformance(Performance):
    def __init__(
        self,
    ) -> None:
        super().__init__("mps")

    def _process(self) -> None:
        pass

    @property
    def utilization(self) -> float:
        return float("nan")

    @property
    def temperature(self) -> float:
        return float("nan")

    @property
    def total_memory(self) -> int:
        return 0

    @property
    def free_memory(self) -> int:
        return 0

    @property
    def used_memory(self) -> int:
        return 0

    @property
    def memory_percent(self) -> float:
        return 0.0
