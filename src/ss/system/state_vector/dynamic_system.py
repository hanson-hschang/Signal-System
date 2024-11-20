from typing import Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ss.system.state_vector import System
from ss.tool.assertion import isPositiveNumber
from ss.tool.assertion.validator import Validator
from ss.tool.descriptor import TensorDescriptor


class ContinuousTimeSystem(System):
    class _NoiseCovarianceValidator(Validator):
        def __init__(
            self,
            dimension: int,
            name: Optional[str] = None,
            noise_covariance: Optional[ArrayLike] = None,
        ) -> None:
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
        self._time_step = time_step

        super().__init__(
            state_dim=state_dim,
            observation_dim=observation_dim,
            control_dim=control_dim,
            number_of_systems=number_of_systems,
            **kwargs,
        )

        self._process_noise_covariance = self._NoiseCovarianceValidator(
            state_dim, "process_noise_covariance", process_noise_covariance
        ).get_noise_covariance()
        self._observation_noise_covariance = self._NoiseCovarianceValidator(
            observation_dim,
            "observation_noise_covariance",
            observation_noise_covariance,
        ).get_noise_covariance()

    process_noise_covariance = TensorDescriptor("_state_dim", "_state_dim")
    observation_noise_covariance = TensorDescriptor(
        "_observation_dim", "_observation_dim"
    )

    def create_multiple_systems(
        self, number_of_systems: int
    ) -> "ContinuousTimeSystem":
        """
        Create multiple systems based on the current system.

        Parameters
        ----------
        `number_of_systems: int`
            The number of systems to be created.

        Returns
        -------
        `system: ContinuousTimeSystem`
            The created multi-system.
        """
        return self.__class__(
            time_step=self._time_step,
            state_dim=self._state_dim,
            observation_dim=self._observation_dim,
            control_dim=self._control_dim,
            number_of_systems=number_of_systems,
            process_noise_covariance=self._process_noise_covariance,
            observation_noise_covariance=self._observation_noise_covariance,
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
        self._update(
            self._state,
            self._compute_state_process(),
            self._compute_process_noise(),
        )
        return time + self._time_step

    def _compute_process_noise(self) -> NDArray[np.float64]:
        process_noise = np.random.multivariate_normal(
            np.zeros(self._state_dim),
            self._process_noise_covariance * np.sqrt(self._time_step),
            size=self._number_of_systems,
        )
        return process_noise

    def _compute_observation_noise(self) -> NDArray[np.float64]:
        observation_noise = np.random.multivariate_normal(
            np.zeros(self._observation_dim),
            self._observation_noise_covariance * np.sqrt(self._time_step),
            size=self._number_of_systems,
        )
        return observation_noise


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

    def create_multiple_systems(
        self, number_of_systems: int
    ) -> "DiscreteTimeSystem":
        """
        Create multiple systems based on the current system.

        Parameters
        ----------
        `number_of_systems: int`
            The number of systems to be created.

        Returns
        -------
        `system: DiscreteTimeSystem`
            The created multi-system.
        """
        return self.__class__(
            state_dim=self._state_dim,
            observation_dim=self._observation_dim,
            control_dim=self._control_dim,
            number_of_systems=number_of_systems,
            process_noise_covariance=self._process_noise_covariance,
            observation_noise_covariance=self._observation_noise_covariance,
        )
