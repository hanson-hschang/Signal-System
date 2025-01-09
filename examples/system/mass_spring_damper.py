from typing import assert_never

import click
import numpy as np

from ss.system.examples.mass_spring_damper import (
    ControlChoice,
    MassSpringDamperSystem,
    ObservationChoice,
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
            observation_noise_covariance = np.diag(
                [observation_noise_variance]
            )
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
