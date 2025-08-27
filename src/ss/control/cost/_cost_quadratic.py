from typing import Optional

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ss.utility.assertion.validator import Validator
from ss.utility.descriptor import NDArrayDescriptor

from ._cost import Cost


class QuadraticCost(Cost):
    class _CostWeightValidator(Validator):
        def __init__(self, cost_weight: ArrayLike) -> None:
            super().__init__(cost_weight)
            self._cost_weight = np.array(cost_weight)
            self._validate_functions.append(self._validate_shape)

        def _validate_shape(self) -> bool:
            shape = self._cost_weight.shape
            if (len(shape) == 2) and (shape[0] == shape[1]):
                return True
            self._errors.append("cost_weight should be a square matrix")
            return False

        def get_weight(self) -> NDArray:
            return self._cost_weight

    class _TerminalCostWeightValidator(Validator):
        def __init__(
            self, dimension: int, cost_weight: Optional[ArrayLike] = None
        ) -> None:
            super().__init__(cost_weight)
            if cost_weight is None:
                cost_weight = np.zeros((dimension, dimension))
            self._cost_weight = np.array(cost_weight)
            self._dimension = dimension
            self._validate_functions.append(self._validate_shape)

        def _validate_shape(self) -> bool:
            shape = self._cost_weight.shape
            if (len(shape) == 2) and (shape[0] == shape[1]):
                return True
            self._errors.append("cost_weight should be a square matrix")
            return False

        def get_weight(self) -> NDArray:
            return self._cost_weight

    class _IntrinsicVectorValidator(Validator):
        def __init__(
            self,
            intrinsic_vector: Optional[ArrayLike],
            dimension: int,
        ) -> None:
            super().__init__(intrinsic_vector)
            if intrinsic_vector is None:
                intrinsic_vector = np.zeros(dimension)
            self._intrinsic_vector = np.array(intrinsic_vector)
            self._dimension = dimension
            self._validate_functions.append(self._validate_shape)

        def _validate_shape(self) -> bool:
            shape = self._intrinsic_vector.shape
            if (len(shape) == 1) and (shape[0] == self._dimension):
                return True
            self._errors.append(
                f"length of intrinsic_vector should be equal to dimension {self._dimension}"
            )
            return False

        def get_vector(self) -> NDArray:
            return self._intrinsic_vector

    def __init__(
        self,
        running_cost_state_weight: NDArray,
        running_cost_control_weight: NDArray,
        terminal_cost_state_weight: Optional[NDArray] = None,
        terminal_cost_control_weight: Optional[NDArray] = None,
        intrinsic_state: Optional[NDArray] = None,
        intrinsic_control: Optional[NDArray] = None,
        time_step: float = 1.0,
        batch_size: int = 1,
    ) -> None:
        self._running_cost_state_weight = self._CostWeightValidator(
            running_cost_state_weight
        ).get_weight()
        state_dim = self._running_cost_state_weight.shape[0]
        self._running_cost_control_weight = self._CostWeightValidator(
            running_cost_control_weight
        ).get_weight()
        control_dim = self._running_cost_control_weight.shape[0]

        self._terminal_cost_state_weight = self._TerminalCostWeightValidator(
            state_dim, terminal_cost_state_weight
        ).get_weight()
        self._terminal_cost_control_weight = self._TerminalCostWeightValidator(
            control_dim, terminal_cost_control_weight
        ).get_weight()

        self._intrinsic_state = self._IntrinsicVectorValidator(
            intrinsic_state,
            state_dim,
        ).get_vector()
        self._intrinsic_control = self._IntrinsicVectorValidator(
            intrinsic_control,
            control_dim,
        ).get_vector()

        super().__init__(
            time_step=time_step,
            state_dim=state_dim,
            control_dim=control_dim,
            batch_size=batch_size,
        )

    running_cost_state_weight = NDArrayDescriptor("_state_dim", "_state_dim")
    running_cost_control_weight = NDArrayDescriptor(
        "_control_dim", "_control_dim"
    )
    terminal_cost_state_weight = NDArrayDescriptor("_state_dim", "_state_dim")
    terminal_cost_control_weight = NDArrayDescriptor(
        "_control_dim", "_control_dim"
    )
    intrinsic_state = NDArrayDescriptor("_state_dim")
    intrinsic_control = NDArrayDescriptor("_control_dim")

    def duplicate(self, batch_size: int) -> "QuadraticCost":
        """
        Create multiple costs.

        Parameters
        ----------
        `batch_size: int`
            The number of costs to be created.

        Returns
        -------
        `cost: QuadraticCost`
            The created multi-cost.
        """
        return self.__class__(
            running_cost_state_weight=self._running_cost_state_weight,
            running_cost_control_weight=self._running_cost_control_weight,
            terminal_cost_state_weight=self._terminal_cost_state_weight,
            terminal_cost_control_weight=self._terminal_cost_control_weight,
            intrinsic_state=self._intrinsic_state,
            intrinsic_control=self._intrinsic_control,
            time_step=self._time_step,
            batch_size=batch_size,
        )

    def _evaluate_running(self) -> None:
        self._compute_cost(
            self._cost,
            self._state,
            self._control,
            self._running_cost_state_weight * self._time_step,
            self._running_cost_control_weight * self._time_step,
            self._intrinsic_state,
            self._intrinsic_control,
        )

    def _evaluate_terminal(self) -> None:
        self._compute_cost(
            self._cost,
            self._state,
            self._control,
            self._terminal_cost_state_weight,
            self._terminal_cost_control_weight,
            self._intrinsic_state,
            self._intrinsic_control,
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _compute_cost(
        cost: NDArray,
        state: NDArray,
        control: NDArray,
        running_cost_state_weight: NDArray,
        running_cost_control_weight: NDArray,
        intrinsic_state: NDArray,
        intrinsic_control: NDArray,
    ) -> None:
        for i in range(cost.shape[0]):
            delta_state = state[i, :] - intrinsic_state
            delta_control = control[i, :] - intrinsic_control
            cost[i] = 0.5 * (
                delta_state @ running_cost_state_weight @ delta_state.T
                + delta_control @ running_cost_control_weight @ delta_control.T
            )
