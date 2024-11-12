from typing import Optional, assert_never

from enum import StrEnum

import click
import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import expm

from system.dense_state.linear import DiscreteTimeLinearSystem
from tool.assertion import isPositiveInteger


class ObservationChoice(StrEnum):
    ALL_STATES = "ALL_STATES"
    ALL_POSITIONS = "ALL_POSITIONS"
    LAST_POSITION = "LAST_POSITION"


class ControlChoice(StrEnum):
    ALL_FORCES = "ALL_FORCES"
    LAST_FORCE = "LAST_FORCE"
    NO_CONTROL = "NO_CONTROL"


class MassSpringDamperSystem(DiscreteTimeLinearSystem):
    def __init__(
        self,
        number_of_connections: int = 1,
        mass: float = 1.0,
        spring_constant: float = 1.0,
        damping_coefficient: float = 1.0,
        time_step: float = 0.01,
        observation_choice: ObservationChoice = ObservationChoice.LAST_POSITION,
        control_choice: ControlChoice = ControlChoice.NO_CONTROL,
        process_noise_covariance: Optional[ArrayLike] = None,
        observation_noise_covariance: Optional[ArrayLike] = None,
        number_of_systems: int = 1,
    ) -> None:
        assert isPositiveInteger(
            number_of_connections
        ), "number_of_connections should be a positive integer"
        assert isinstance(
            observation_choice, ObservationChoice
        ), f"observation_choice should be one of {ObservationChoice.__members__.keys()}"
        assert isinstance(
            control_choice, ControlChoice
        ), f"control_choice should be one of {ControlChoice.__members__.keys()}"

        self._number_of_connections = int(number_of_connections)
        self._mass = mass
        self._spring = spring_constant
        self._damping = damping_coefficient

        matrix_A = np.zeros(
            (2 * self._number_of_connections, 2 * self._number_of_connections)
        )
        matrix_A[
            : self._number_of_connections, self._number_of_connections :
        ] = np.identity(self._number_of_connections)
        matrix_A[self._number_of_connections, 0] = -self._spring / self._mass
        matrix_A[self._number_of_connections, self._number_of_connections] = (
            -self._damping / self._mass
        )
        for i in range(1, self._number_of_connections):
            position_index = i - 1
            velocity_index = self._number_of_connections + i - 1
            matrix_A[
                velocity_index + 0, position_index : position_index + 2
            ] += (np.array([-1, 1]) * self._spring / self._mass)
            matrix_A[
                velocity_index + 1, position_index : position_index + 2
            ] -= (np.array([-1, 1]) * self._spring / self._mass)
            matrix_A[
                velocity_index + 0, velocity_index : velocity_index + 2
            ] += (np.array([-1, 1]) * self._damping / self._mass)
            matrix_A[
                velocity_index + 1, velocity_index : velocity_index + 2
            ] -= (np.array([-1, 1]) * self._damping / self._mass)

        self._observation_choice = observation_choice
        match self._observation_choice:
            case ObservationChoice.ALL_STATES:
                matrix_C = np.identity(2 * self._number_of_connections)
            case ObservationChoice.ALL_POSITIONS:
                matrix_C = np.zeros(
                    (
                        self._number_of_connections,
                        2 * self._number_of_connections,
                    )
                )
                matrix_C[:, self._number_of_connections :] = np.identity(
                    self._number_of_connections
                )
            case ObservationChoice.LAST_POSITION:
                matrix_C = np.zeros((1, 2 * self._number_of_connections))
                matrix_C[0, -1] = 1
            case _ as unmatched_observation_choice:  # pragma: no cover
                assert_never(unmatched_observation_choice)

        self._control_choice = control_choice
        match self._control_choice:
            case ControlChoice.ALL_FORCES:
                matrix_B = np.zeros(
                    (
                        2 * self._number_of_connections,
                        self._number_of_connections,
                    )
                )
                matrix_B[self._number_of_connections :, :] = (
                    np.identity(self._number_of_connections) / self._mass
                )
            case ControlChoice.LAST_FORCE:
                matrix_B = np.zeros((2 * self._number_of_connections, 1))
                matrix_B[-1, 0] = 1 / self._mass
            case ControlChoice.NO_CONTROL:
                matrix_B = np.zeros((2 * self._number_of_connections, 0))
            case _ as unmatched_control_choice:
                assert_never(unmatched_control_choice)

        process_noise_covariance = np.array(process_noise_covariance)
        observation_noise_covariance = np.array(observation_noise_covariance)

        state_space_matrix_A = expm(matrix_A * time_step)
        state_space_matrix_B = state_space_matrix_A @ matrix_B * time_step

        super().__init__(
            state_space_matrix_A=state_space_matrix_A,
            state_space_matrix_B=state_space_matrix_B,
            state_space_matrix_C=matrix_C,
            process_noise_covariance=process_noise_covariance,
            observation_noise_covariance=observation_noise_covariance,
            number_of_systems=number_of_systems,
        )
        self._time_step = time_step

    def create_multiple_systems(
        self, number_of_systems: int
    ) -> "MassSpringDamperSystem":
        """
        Create multiple systems based on the current system.

        Parameters
        ----------
        `number_of_systems: int`
            The number of systems to be created.

        Returns
        -------
        `system: MassSpringDamperSystem`
            The created multi-system.
        """
        return self.__class__(
            number_of_connections=self._number_of_connections,
            mass=self._mass,
            spring_constant=self._spring,
            damping_coefficient=self._damping,
            time_step=self._time_step,
            observation_choice=self._observation_choice,
            control_choice=self._control_choice,
            process_noise_covariance=self.process_noise_covariance,
            observation_noise_covariance=self.observation_noise_covariance,
            number_of_systems=number_of_systems,
        )


@click.command()
@click.option(
    "--number-of-connections",
    type=click.IntRange(min=1),
    default=2,
    help="Set the number of connections (positive integers).",
)
@click.option(
    "--damping-coefficient",
    type=click.FloatRange(min=0),
    default=0.0,
    help="Set the damping coefficient (non-negative value).",
)
@click.option(
    "--time-step",
    type=click.FloatRange(min=0),
    default=0.01,
    help="Set the time step (positive value).",
)
@click.option(
    "--number-of-systems",
    type=click.IntRange(min=1),
    default=1,
    help="Set the number of systems (positive integers).",
)
@click.option(
    "--process-noise-variance",
    type=click.FloatRange(min=0),
    default=0.01,
    help="Set the process noise variance (non-negative value).",
)
@click.option(
    "--observation-noise-variance",
    type=click.FloatRange(min=0),
    default=0.01,
    help="Set the observation noise variance (non-negative value).",
)
@click.option(
    "--observation-choice",
    type=click.Choice(
        list(ObservationChoice.__members__.keys()),
        case_sensitive=False,
    ),
    default=ObservationChoice.ALL_POSITIONS,
    help="Set the observation choice.",
)
@click.option(
    "--control-choice",
    type=click.Choice(
        list(ControlChoice.__members__.keys()),
        case_sensitive=False,
    ),
    default=ControlChoice.NO_CONTROL,
    help="Set the control choice.",
)
def main(
    number_of_connections: int,
    damping_coefficient: float,
    time_step: float,
    number_of_systems: int,
    process_noise_variance: float,
    observation_noise_variance: float,
    observation_choice: str,
    control_choice: str,
) -> None:
    control_choice = ControlChoice(control_choice)
    observation_choice = ObservationChoice(observation_choice)
    match observation_choice:
        case ObservationChoice.ALL_STATES:
            observation_noise_covariance = np.diag(
                [observation_noise_variance] * 2 * number_of_connections
            )
        case ObservationChoice.ALL_POSITIONS:
            observation_noise_covariance = np.diag(
                [observation_noise_variance] * number_of_connections
            )
        case ObservationChoice.LAST_POSITION:
            observation_noise_covariance = np.diag([observation_noise_variance])
        case _ as unmatched_observation_choice:
            assert_never(unmatched_observation_choice)
    process_noise_covariance = process_noise_variance * np.eye(
        2 * number_of_connections
    )

    linear_system = MassSpringDamperSystem(
        number_of_connections=number_of_connections,
        damping_coefficient=damping_coefficient,
        time_step=time_step,
        observation_choice=observation_choice,
        control_choice=control_choice,
        process_noise_covariance=process_noise_covariance,
        observation_noise_covariance=observation_noise_covariance,
        number_of_systems=number_of_systems,
    )
    print(linear_system.state.shape)
    linear_system.process(0)
    observation = linear_system.observe()
    print(observation)


if __name__ == "__main__":
    main()
