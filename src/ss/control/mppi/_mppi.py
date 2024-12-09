from typing import Optional

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ss.control.cost.quadratic import QuadraticCost
from ss.signal.smoothing.moving_averaging import MovingAveragingSmoother
from ss.system import ContinuousTimeSystem
from ss.utility.assertion import (
    is_nonnegative_number,
    is_positive_integer,
    is_positive_number,
)
from ss.utility.descriptor import ReadOnlyDescriptor, TensorDescriptor


class ModelPredictivePathIntegralController:
    """
    Model Predictive Path Integral Controller [Williams et al., 2018]

    Parameters:
    -----------
    system: ContinuousTimeSystem
        A continuous-time dynamical system
    cost: QuadraticCost
        A quadratic cost function
    time_horizon: int
        Time horizon
    number_of_samples: int
        Number of samples
    temperature: float
        Temperature parameter
    smoothing_window_size: int
        Smoothing window size

    References:
    -----------
    arXiv version: https://arxiv.org/abs/1707.02342
    IEEE Transactions on Robotics version: https://ieeexplore.ieee.org/ielaam/8860/8558659/8558663-aam.pdf
    """

    def __init__(
        self,
        system: ContinuousTimeSystem,
        cost: QuadraticCost,
        time_horizon: int,
        number_of_samples: int,
        temperature: float,
        base_control_confidence: float,
        exploration_percentage: float = 0.0,
        smoothing_window_size: Optional[int] = None,
    ) -> None:
        assert issubclass(
            type(system), ContinuousTimeSystem
        ), f"system {system} must be an instance of ContinuousTimeSystem"
        assert issubclass(
            type(cost), QuadraticCost
        ), f"cost {cost} must be an instance of QuadraticCost"

        assert (
            system.state_dim == cost.state_dim
        ), f"state_dim of system {system.state_dim} must match with the state_dim of cost {cost.state_dim}"
        assert (
            system.control_dim == cost.control_dim
        ), f"control_dim of system {system.control_dim} must match with the control_dim of cost {cost.control_dim}"

        assert is_positive_integer(
            time_horizon
        ), f"time_horizon {time_horizon} must be a positive integer"
        assert is_positive_integer(
            number_of_samples
        ), f"number_of_samples {number_of_samples} must be a positive integer"
        assert is_positive_number(
            temperature
        ), f"temperature {temperature} must be a positive number"
        assert (
            is_positive_number(base_control_confidence)
            and base_control_confidence <= 1
        ), f"base_control_confidence {base_control_confidence} must be a positive number within the range (0, 1]"
        assert (
            is_nonnegative_number(exploration_percentage)
            and exploration_percentage < 1
        ), f"exploration_percentage {exploration_percentage} must be a positive number within the range (0, 1)"
        if smoothing_window_size is None:
            smoothing_window_size = int(time_horizon * 0.1) + 1
        assert is_positive_integer(
            smoothing_window_size
        ), f"smoothing_window_size {smoothing_window_size} must be a positive integer"

        self._systems: ContinuousTimeSystem = system.create_multiple_systems(
            number_of_samples
        )
        self._control_dim: int = system.control_dim
        self._costs: QuadraticCost = cost.create_multiple_costs(
            number_of_samples
        )

        self._time_horizon = time_horizon
        self._number_of_samples = number_of_samples
        self._control_trajectory = np.zeros(
            (self._control_dim, self._time_horizon)
        )
        self._temperature = temperature
        self._base_control_confidence = base_control_confidence
        self._exploration_percentage = exploration_percentage
        self._exploration_index = self._compute_exploration_index()
        self._control_cost_regularization_weight = (
            self._compute_control_cost_regularization_weight()
        )
        self._smoothing_window_size = smoothing_window_size

        self._moving_average_smoother = MovingAveragingSmoother(
            self._smoothing_window_size
        )

    time_horizon = ReadOnlyDescriptor[int]()
    number_of_samples = ReadOnlyDescriptor[int]()
    control_dim = ReadOnlyDescriptor[int]()
    smoothing_window_size = ReadOnlyDescriptor[int]()
    control_trajectory = TensorDescriptor("_control_dim", "_time_horizon")

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, temperature: float) -> None:
        assert is_positive_number(
            temperature
        ), f"temperature {temperature} must be a positive number"
        self._temperature = temperature
        self._control_cost_regularization_weight = (
            self._compute_control_cost_regularization_weight()
        )

    @property
    def base_control_confidence(self) -> float:
        return self._base_control_confidence

    @base_control_confidence.setter
    def base_control_confidence(self, base_control_confidence: float) -> None:
        assert (
            is_positive_number(base_control_confidence)
            and base_control_confidence <= 1
        ), f"base_control_confidence {base_control_confidence} must be a positive number within the range (0, 1]"
        self._base_control_confidence = base_control_confidence
        self._control_cost_regularization_weight = (
            self._compute_control_cost_regularization_weight()
        )

    def _compute_exploration_index(self) -> int:
        return int(
            (1 - self._base_control_confidence) * self._number_of_samples
        )

    def _compute_control_cost_regularization_weight(self) -> float:
        return (1 - self._base_control_confidence) * self._temperature

    def reset_systems(self, state: Optional[ArrayLike] = None) -> None:
        if state is None:
            state = np.zeros_like(self._systems._state_dim)
        state = np.array(state, dtype=np.float64).squeeze()
        assert (len(state.shape) == 1) and (
            state.shape[0] == self._systems._state_dim
        ), f"state {state} must be a vector with length {self._systems._state_dim}"
        self._systems.state = np.tile(state, (self._number_of_samples, 1))

    def compute_control(self) -> NDArray[np.float64]:
        """
        Algorithm 1 of [Williams et al., 2018]
        """
        control_trajectory = self._reinitialize_control_trajectory()
        total_cost = np.zeros(self._number_of_samples)
        noisy_exploration_control_trajectory = (
            self._compute_noisy_exploration_control_trajectory()
        )

        time = 0.0
        for k in range(self._time_horizon):

            self._costs.state = self._systems.state

            base_control = control_trajectory[:, k][np.newaxis, :]

            control = (
                noisy_exploration_control_trajectory[:, k, :] + base_control
            )
            # control = noisy_exploration_control_trajectory[:, k, :]
            # control[:self._exploration_index, :] += base_control

            self._systems.control = control
            time = self._systems.process(time)

            total_cost += self._costs.evaluate() + (
                self._control_cost_regularization_weight
                * np.einsum(
                    "m, im -> i",
                    self._costs.running_cost_control_weight
                    @ base_control[0, :],
                    control,
                )
                * self._costs.time_step
            )  # TODO: implement this in numba
        self._costs.set_terminal()
        total_cost += self._costs.evaluate()

        weight = self._compute_weight(total_cost, self._temperature)

        exploration_control_trajectory = (
            self._compute_exploration_control_trajectory(
                weight,
                noisy_exploration_control_trajectory,
            )
        )

        exploration_control_trajectory = self._moving_average_smoother.smooth(
            exploration_control_trajectory
        )

        self._update_control_trajectory(
            self._control_trajectory,
            exploration_control_trajectory,
        )

        return self._control_trajectory

    def _reinitialize_control_trajectory(self) -> NDArray[np.float64]:
        control_trajectory = np.zeros_like(self._control_trajectory)
        control_trajectory[:, :-1] = self._control_trajectory[:, 1:]
        control_trajectory[:, -1] = self._control_trajectory[:, -1]
        return control_trajectory

    def _compute_noisy_exploration_control_trajectory(
        self,
    ) -> NDArray[np.float64]:
        noisy_exploration_control_trajectory = np.random.multivariate_normal(
            np.zeros(self._control_dim),
            np.linalg.inv(self._costs.running_cost_control_weight),
            size=(self._number_of_samples, self._time_horizon),
        )
        return noisy_exploration_control_trajectory

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _compute_weight(
        total_cost: NDArray[np.float64],
        temperature: float,
    ) -> NDArray[np.float64]:
        """
        Algorithm 2 of [Williams et al., 2018]
        """
        weight: NDArray[np.float64] = np.exp(
            -1.0 / temperature * (total_cost - np.min(total_cost))
        )
        weight /= np.sum(weight)
        return weight

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _compute_exploration_control_trajectory(
        weight: NDArray[np.float64],
        noisy_exploration_control_trajectory: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        exploration_control_trajectory = np.zeros(
            noisy_exploration_control_trajectory.shape[-1:0:-1]
        )
        for k in range(noisy_exploration_control_trajectory.shape[2]):
            exploration_control_trajectory[:, k] = np.sum(
                noisy_exploration_control_trajectory[:, k, :]
                * weight[:, np.newaxis],
                axis=0,
            )
        return exploration_control_trajectory

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _update_control_trajectory(
        control_trajectory: NDArray[np.float64],
        exploration_control_trajectory: NDArray[np.float64],
    ) -> None:
        control_trajectory[:, :] += exploration_control_trajectory
