from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_continuous_are, solve_discrete_are

from ss.control.cost.quadratic import QuadraticCost
from ss.system.finite_state.linear import (
    ContinuousTimeLinearSystem,
    DiscreteTimeLinearSystem,
)


class LQGController:
    def __init__(
        self,
        system: Union[ContinuousTimeLinearSystem, DiscreteTimeLinearSystem],
        cost: QuadraticCost,
    ) -> None:
        assert isinstance(
            system, (ContinuousTimeLinearSystem, DiscreteTimeLinearSystem)
        ), f"system {system} must be an instance of either ContinuousTimeLinearSystem or DiscreteTimeLinearSystem"
        assert isinstance(
            cost, QuadraticCost
        ), f"cost {cost} must be an instance of QuadraticCost"

        self._system = system
        self._cost = cost

        if isinstance(system, ContinuousTimeLinearSystem):
            solve_are = solve_continuous_are
        elif isinstance(system, DiscreteTimeLinearSystem):
            solve_are = solve_discrete_are
        else:
            raise ValueError("Invalid system type")

        self._are_solution = solve_are(
            a=self._system.state_space_matrix_A,
            b=self._system.state_space_matrix_B,
            q=self._cost.running_cost_state_weight,
            r=self._cost.running_cost_control_weight,
        )

        self._optimal_gain_inf = -(
            self._cost.running_cost_control_weight
            @ self._system.state_space_matrix_B.T
            @ self._are_solution
        )

    def compute_control(self) -> NDArray[np.float64]:
        state = self._system.state
        if self._system.number_of_systems == 1:
            state = state[np.newaxis, ...]
        control = self._compute_control(
            self._optimal_gain_inf,
            state,
        ).squeeze()
        return control

    def _compute_control(
        self,
        gain: NDArray[np.float64],
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        number_of_systems = state.shape[0]
        control = np.zeros((number_of_systems, gain.shape[0]))
        for i in range(number_of_systems):
            control[i, :] = gain @ state[i, :]
        return control
