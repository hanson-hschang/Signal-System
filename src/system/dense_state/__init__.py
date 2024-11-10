from typing import Any, Optional, Union

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from assertion import isNonNegativeInteger, isPositiveInteger, isPositiveNumber
from assertion.validator import Validator
from tool.descriptor import ReadOnlyDescriptor, TensorDescriptor


class ContinuousTimeSystem:
    class _NoiseCovarianceValidator(Validator):
        def __init__(
            self,
            dimension: int,
            name: Optional[str] = None,
            noise_covariance: Optional[ArrayLike] = None,
        ):
            super().__init__()
            if noise_covariance is None:
                noise_covariance = np.zeros((dimension, dimension))
            self._noise_covariance = np.array(
                noise_covariance, dtype=np.float64
            )
            self._dimension = dimension
            self._name = name if name is not None else "noise_covariance"
            self._validate_shape()

        def _validate_shape(self) -> None:
            shape = self._noise_covariance.shape
            if not (
                len(shape) == 2 and (shape[0] == shape[1] == self._dimension)
            ):
                self._errors.append(
                    self._name
                    + f" should be a square matrix and have shape ({self._dimension}, {self._dimension})"
                )

        def get_noise_covariance(self) -> NDArray[np.float64]:
            return self._noise_covariance

    def __init__(
        self,
        time_step: Union[int, float],
        state_dim: int,
        observation_dim: int,
        control_dim: int = 0,
        number_of_systems: int = 1,
        process_noise_covariance: Optional[ArrayLike] = None,
        observation_noise_covariance: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> None:
        assert isPositiveNumber(
            time_step
        ), f"time_step {time_step} must be a positive number"
        assert isPositiveInteger(
            state_dim
        ), f"state_dim {state_dim} must be a positive integer"
        assert isPositiveInteger(
            observation_dim
        ), f"observation_dim {observation_dim} must be a positive integer"
        assert isNonNegativeInteger(
            control_dim
        ), f"control_dim {control_dim} must be a non-negative integer"
        assert isPositiveInteger(
            number_of_systems
        ), f"number_of_systems {number_of_systems} must be a positive integer"

        self._process_noise_covariance = self._NoiseCovarianceValidator(
            state_dim, "process_noise_covariance", process_noise_covariance
        ).get_noise_covariance()
        self._observation_noise_covariance = self._NoiseCovarianceValidator(
            observation_dim,
            "observation_noise_covariance",
            observation_noise_covariance,
        ).get_noise_covariance()

        self._time_step = time_step
        self._state_dim = int(state_dim)
        self._observation_dim = int(observation_dim)
        self._control_dim = int(control_dim)
        self._number_of_systems = int(number_of_systems)
        self._state = np.zeros(
            (self._number_of_systems, self._state_dim), dtype=np.float64
        )
        self._observation = np.zeros(
            (self._number_of_systems, self._observation_dim), dtype=np.float64
        )
        self._control = np.zeros(
            (self._number_of_systems, self._control_dim), dtype=np.float64
        )
        super().__init__(**kwargs)

    state_dim = ReadOnlyDescriptor[int]()
    observation_dim = ReadOnlyDescriptor[int]()
    control_dim = ReadOnlyDescriptor[int]()
    number_of_systems = ReadOnlyDescriptor[int]()
    state = TensorDescriptor("_number_of_systems", "_state_dim")
    control = TensorDescriptor("_number_of_systems", "_control_dim")
    process_noise_covariance = TensorDescriptor("_state_dim", "_state_dim")
    observation_noise_covariance = TensorDescriptor(
        "_observation_dim", "_observation_dim"
    )

    def process(self, time: Union[int, float]) -> Union[int, float]:
        """
        Update the state of each system by one time step based on the current state and control (if existed).

        Parameters
        ----------
        `time: Union[int, float]`
            The current time.

        Returns
        -------
        `time: Union[int, float]`
            The updated time.
        """
        process_noise = np.random.multivariate_normal(
            np.zeros(self._state_dim),
            self._process_noise_covariance * np.sqrt(self._time_step),
            size=self._number_of_systems,
        )
        self._update(
            self._state,
            self._compute_state_process(),
            process_noise,
        )
        return time + self._time_step

    def observe(self) -> NDArray[np.float64]:
        """
        Make observation of each system based on the current state.

        Returns
        -------
        `observation: ArrayLike[float]`
            The observation vector of systems. Shape of the array is `(number_of_systems, observation_dim)`.
        """
        observation_noise = np.random.multivariate_normal(
            np.zeros(self._observation_dim),
            self._observation_noise_covariance * np.sqrt(self._time_step),
            size=self._number_of_systems,
        )
        self._update(
            self._observation,
            self._compute_observation_process(),
            observation_noise,
        )
        observation: NDArray[np.float64] = (
            self._observation[0]
            if self._number_of_systems == 1
            else self._observation
        )
        return observation

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _update(
        array: NDArray[np.float64],
        process: NDArray[np.float64],
        noise: NDArray[np.float64],
    ) -> None:
        array[:, :] = process + noise

    def _compute_state_process(self) -> NDArray[np.float64]:
        return self._state

    def _compute_observation_process(self) -> NDArray[np.float64]:
        return np.zeros_like(self._observation)


class DiscreteTimeSystem(ContinuousTimeSystem):
    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        control_dim: int = 0,
        number_of_systems: int = 1,
        process_noise_covariance: Optional[ArrayLike] = None,
        observation_noise_covariance: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            time_step=1,
            state_dim=state_dim,
            observation_dim=observation_dim,
            control_dim=control_dim,
            number_of_systems=number_of_systems,
            process_noise_covariance=process_noise_covariance,
            observation_noise_covariance=observation_noise_covariance,
            **kwargs,
        )
