import numpy as np

from ss.system.finite_state import System


class Estimator:

    def __init__(self, system: System) -> None:
        assert issubclass(
            type(system), System
        ), "system must be a subclass of System"
        self._system = system
        self._state = np.zeros_like(system.state)
