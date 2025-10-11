from collections.abc import Callable
from typing import assert_never

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.linalg import solve_continuous_are, solve_discrete_are

from ss.control import Controller
from ss.control.cost import QuadraticCost
from ss.system.linear import (
    ContinuousTimeLinearSystem,
    DiscreteTimeLinearSystem,
)
from ss.utility.assertion import is_nonnegative_integer
from ss.utility.descriptor import BatchNDArrayDescriptor, ReadOnlyDescriptor


class LinearQuadraticRegulatorController(Controller):
    def __init__(
        self,
        system: ContinuousTimeLinearSystem | DiscreteTimeLinearSystem,
        cost: QuadraticCost,
        time_horizon: int = 0,
    ) -> None:
        assert isinstance(
            system, (ContinuousTimeLinearSystem, DiscreteTimeLinearSystem)
        ), (
            f"{type(system) = } must be an instance of either "
            "ContinuousTimeLinearSystem or DiscreteTimeLinearSystem"
        )
        assert isinstance(cost, QuadraticCost), (
            f"{type(cost) = } must be an instance of QuadraticCost"
        )

        assert system.state_dim == cost.state_dim, (
            f"{system.state_dim = } must be the same as {cost.state_dim = }"
        )
        assert system.control_dim == cost.control_dim, (
            f"{system.control_dim = } must be "
            f"the same as {cost.control_dim = }"
        )

        assert is_nonnegative_integer(time_horizon), (
            f"{time_horizon = } must be a nonnegative integer"
        )

        super().__init__(
            control_dim=system.control_dim,
            batch_size=system.batch_size,
        )

        self._time_horizon = time_horizon
        self._state_dim = system.state_dim
        self._system_state = np.zeros_like(system.state)

        self._optimal_gain = np.zeros(
            (self._control_dim, self._state_dim),
            dtype=np.float64,
        )

        if self._time_horizon == 0:
            # Infinite time horizon case:
            # The optimal gain is the negative of
            # running_cost_control_weight @ B @ solution_are
            # where solution_are is the solution to the
            # algebraic Riccati equation (ARE)
            if isinstance(system, ContinuousTimeLinearSystem):
                solve_are = solve_continuous_are
            elif isinstance(system, DiscreteTimeLinearSystem):
                solve_are = solve_discrete_are
            else:
                assert_never(system)
            solution_are = solve_are(
                a=system.state_space_matrix_A,
                b=system.state_space_matrix_B,
                q=cost.running_cost_state_weight,
                r=cost.running_cost_control_weight,
            )
            self._optimal_gain = -(
                cost.running_cost_control_weight
                @ system.state_space_matrix_B.T
                @ solution_are
            )
            self._compute_control: Callable = (
                self._compute_infinite_horizon_optimal_control
            )
        else:
            # Finite time horizon case:
            # TODO: Implement finite horizon LQR
            pass

    time_horizon = ReadOnlyDescriptor[int]()
    system_state = BatchNDArrayDescriptor(
        "_batch_size",
        "_state_dim",
    )

    def _compute_infinite_horizon_optimal_control(self) -> NDArray[np.float64]:
        control: NDArray[np.float64] = self._gain_multiply_state(
            self._optimal_gain, self._system_state
        )
        return control

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _gain_multiply_state(
        gain: NDArray[np.float64],
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        batch_size = state.shape[0]
        control = np.zeros((batch_size, gain.shape[0]), dtype=np.float64)
        for i in range(batch_size):
            control[i, :] = gain @ state[i, :]
        return control
