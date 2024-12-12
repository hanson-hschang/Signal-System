from typing import Optional

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ss.control import Controller
from ss.control.cost.quadratic import QuadraticCost
from ss.signal.smoothing.moving_averaging import MovingAveragingSmoother
from ss.system import ContinuousTimeSystem
from ss.utility.assertion import (
    is_nonnegative_number,
    is_positive_integer,
    is_positive_number,
)
from ss.utility.assertion.validator import Validator
from ss.utility.descriptor import (
    MultiSystemTensorDescriptor,
    ReadOnlyDescriptor,
)


class ModelPredictivePathIntegralController(Controller):
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

    class _SystemValidator(Validator):
        def __init__(
            self,
            system: ContinuousTimeSystem,
        ) -> None:
            super().__init__()
            assert issubclass(
                type(system), ContinuousTimeSystem
            ), f"{system = } must be an instance of ContinuousTimeSystem"

    class _CostValidator(Validator):
        def __init__(
            self,
            cost: QuadraticCost,
        ) -> None:
            super().__init__()
            assert issubclass(
                type(cost), QuadraticCost
            ), f"{cost = } must be an instance of QuadraticCost"

    class _DimensionValidator(Validator):
        def __init__(
            self,
            system: ContinuousTimeSystem,
            cost: QuadraticCost,
        ) -> None:
            super().__init__()
            assert (
                system.state_dim == cost.state_dim
            ), f"{system.state_dim = } must be the same as {cost.state_dim = }"
            assert (
                system.control_dim == cost.control_dim
            ), f"{system.control_dim = } must be the same as {cost.control_dim = }"

    class _TimeHorizonValidator(Validator):
        def __init__(
            self,
            time_horizon: int,
        ) -> None:
            super().__init__()
            assert is_positive_integer(
                time_horizon
            ), f"{time_horizon = } must be a positive integer"
            self._time_horizon = time_horizon

        def get_value(self) -> int:
            return self._time_horizon

    class _NumberOfSamplesValidator(Validator):
        def __init__(
            self,
            number_of_samples: int,
        ) -> None:
            super().__init__()
            assert is_positive_integer(
                number_of_samples
            ), f"{number_of_samples = } must be a positive integer"
            self._number_of_samples = number_of_samples

        def get_value(self) -> int:
            return self._number_of_samples

    class _TemperatureValidator(Validator):
        def __init__(
            self,
            temperature: float,
        ) -> None:
            super().__init__()
            assert is_positive_number(
                temperature
            ), f"{temperature = } must be a positive number"
            self._temperature = temperature

        def get_value(self) -> float:
            return self._temperature

    class _BaseControlConfidenceValidator(Validator):
        def __init__(
            self,
            base_control_confidence: float,
        ) -> None:
            super().__init__()
            assert (
                is_positive_number(base_control_confidence)
                and base_control_confidence <= 1
            ), f"{base_control_confidence = } must be a positive number within the range (0, 1]"
            self._base_control_confidence = base_control_confidence

        def get_value(self) -> float:
            return self._base_control_confidence

    class _ExplorationPercentageValidator(Validator):
        def __init__(
            self,
            exploration_percentage: float,
        ) -> None:
            super().__init__()
            assert (
                is_nonnegative_number(exploration_percentage)
                and exploration_percentage < 1
            ), f"{exploration_percentage = } must be a positive number within the range (0, 1)"
            self._exploration_percentage = exploration_percentage

        def get_value(self) -> float:
            return self._exploration_percentage

    class _SmoothingWindowSizeValidator(Validator):
        def __init__(
            self,
            smoothing_window_size: Optional[int],
            time_horizon: int,
        ) -> None:
            super().__init__()
            if smoothing_window_size is None:
                smoothing_window_size = int(time_horizon * 0.1) + 1
            assert is_positive_integer(
                smoothing_window_size
            ), f"{smoothing_window_size = } must be a positive integer"
            self._smoothing_window_size = smoothing_window_size

        def get_value(self) -> int:
            return self._smoothing_window_size

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
        self._SystemValidator(system)
        self._CostValidator(cost)
        self._DimensionValidator(system, cost)
        super().__init__(
            control_dim=system.control_dim,
            number_of_systems=system.number_of_systems,
        )

        self._time_horizon = self._TimeHorizonValidator(
            time_horizon
        ).get_value()
        self._number_of_samples = self._NumberOfSamplesValidator(
            number_of_samples
        ).get_value()
        self._temperature = self._TemperatureValidator(temperature).get_value()
        self._base_control_confidence = self._BaseControlConfidenceValidator(
            base_control_confidence
        ).get_value()
        self._exploration_percentage = self._ExplorationPercentageValidator(
            exploration_percentage
        ).get_value()
        self._smoothing_window_size = self._SmoothingWindowSizeValidator(
            smoothing_window_size, self._time_horizon
        ).get_value()

        self._systems: ContinuousTimeSystem = system.duplicate(
            self._number_of_samples
        )
        self._costs: QuadraticCost = cost.duplicate(self._number_of_samples)

        self._state_dim = system.state_dim
        self._control_dim = system.control_dim
        self._number_of_systems = system.number_of_systems

        self._system_state = np.zeros(
            (self._number_of_systems, self._state_dim),
            dtype=np.float64,
        )
        self._control_trajectory = np.zeros(
            (self._number_of_systems, self._control_dim, self._time_horizon),
            dtype=np.float64,
        )

        self._exploration_index = self._compute_exploration_index()
        self._control_cost_regularization_weight = (
            self._compute_control_cost_regularization_weight()
        )

        self._moving_average_smoother = MovingAveragingSmoother(
            self._smoothing_window_size
        )

    time_horizon = ReadOnlyDescriptor[int]()
    number_of_samples = ReadOnlyDescriptor[int]()
    smoothing_window_size = ReadOnlyDescriptor[int]()
    system_state = MultiSystemTensorDescriptor(
        "_number_of_systems",
        "_state_dim",
    )
    control_trajectory = MultiSystemTensorDescriptor(
        "_number_of_systems",
        "_control_dim",
        "_time_horizon",
    )

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, temperature: float) -> None:
        self._temperature = self._TemperatureValidator(temperature).get_value()
        self._control_cost_regularization_weight = (
            self._compute_control_cost_regularization_weight()
        )

    @property
    def base_control_confidence(self) -> float:
        return self._base_control_confidence

    @base_control_confidence.setter
    def base_control_confidence(self, base_control_confidence: float) -> None:
        self._base_control_confidence = self._BaseControlConfidenceValidator(
            base_control_confidence
        ).get_value()
        self._control_cost_regularization_weight = (
            self._compute_control_cost_regularization_weight()
        )

    def _compute_exploration_index(self) -> int:
        return int(
            (1 - self._base_control_confidence) * self._number_of_samples
        )

    def _compute_control_cost_regularization_weight(self) -> float:
        return (1 - self._base_control_confidence) * self._temperature

    def _reset_systems(self, state: Optional[NDArray] = None) -> None:
        if state is None:
            state = np.zeros_like(self._systems.state_dim)
        self._systems.state = np.tile(state, (self._number_of_samples, 1))

    def _compute_control(self) -> NDArray[np.float64]:
        """
        Algorithm 1 of [Williams et al., 2018]
        """
        control_trajectory = self._reinitialize_control_trajectory()
        noisy_exploration_control_trajectory = (
            self._compute_noisy_exploration_control_trajectory()
        )

        for i in range(self._number_of_systems):
            # reset each sampled system to the initial state
            self._reset_systems(self._system_state[i, :])

            # initialize the total cost
            _total_cost = np.zeros(self._number_of_samples)

            _current_time = 0.0
            for k in range(self._time_horizon):

                # Compute the control
                _base_control = control_trajectory[i, :, k][np.newaxis, :]
                _control = (
                    noisy_exploration_control_trajectory[i, :, :, k]
                    + _base_control
                )
                # control = noisy_exploration_control_trajectory[:, k, :]
                # control[:self._exploration_index, :] += base_control

                # Set the control
                self._systems.control = _control

                # Compute the cost
                self._costs.state = self._systems.state
                _total_cost += self._costs.evaluate() + (
                    self._control_cost_regularization_weight
                    * np.einsum(
                        "m, jm -> j",
                        self._costs.running_cost_control_weight
                        @ _base_control[0, :],
                        _control,
                    )
                    * self._costs.time_step
                )  # TODO: implement this in numba

                # Update the system
                _current_time = self._systems.process(_current_time)

            # Compute the terminal cost
            self._costs.set_terminal()
            self._costs.state = self._systems.state
            _total_cost += self._costs.evaluate()

            # Compute the weight
            _weight = self._compute_weight(_total_cost, self._temperature)

            # Update the control trajectory
            self._update_control_trajectory(
                control_trajectory=self._control_trajectory[i, ...],
                exploration_control_trajectory=self._moving_average_smoother.smooth(
                    self._compute_exploration_control_trajectory(
                        _weight,
                        noisy_exploration_control_trajectory[i, ...],
                    )
                ),
            )

        return self._control_trajectory[:, :, 0]

    def _reinitialize_control_trajectory(self) -> NDArray[np.float64]:
        control_trajectory = np.zeros_like(self._control_trajectory)
        control_trajectory[..., :-1] = self._control_trajectory[..., 1:]
        control_trajectory[..., -1] = self._control_trajectory[..., -1]
        return control_trajectory

    def _compute_noisy_exploration_control_trajectory(
        self,
    ) -> NDArray[np.float64]:
        noisy_exploration_control_trajectory = np.random.multivariate_normal(
            np.zeros(self._control_dim),
            np.linalg.inv(self._costs.running_cost_control_weight),
            size=(
                self._number_of_systems,
                self._number_of_samples,
                self._time_horizon,
            ),
        )
        return noisy_exploration_control_trajectory.transpose(0, 1, 3, 2)

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
            noisy_exploration_control_trajectory.shape[-2:]
        )
        for k in range(noisy_exploration_control_trajectory.shape[-1]):
            exploration_control_trajectory[:, k] = np.sum(
                noisy_exploration_control_trajectory[:, :, k]
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
