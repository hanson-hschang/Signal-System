from typing import assert_never

import click
import numpy as np

from system.linear import MassSpringDamperSystem


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
        MassSpringDamperSystem.ObservationChoice.__members__.keys(),
        case_sensitive=False,
    ),
    default=MassSpringDamperSystem.ObservationChoice.ALL_POSITIONS,
    help="Set the observation choice.",
)
def main(
    number_of_connections: int,
    damping_coefficient: float,
    number_of_systems: int,
    process_noise_variance: float,
    observation_noise_variance: float,
    observation_choice: str,
) -> None:
    observation_choice = MassSpringDamperSystem.ObservationChoice(
        observation_choice
    )
    match observation_choice:
        case MassSpringDamperSystem.ObservationChoice.ALL_POSITIONS:
            observation_noise_covariance = np.diag(
                [observation_noise_variance] * number_of_connections
            )
        case MassSpringDamperSystem.ObservationChoice.LAST_POSITION:
            observation_noise_covariance = np.diag([observation_noise_variance])
        case _:
            assert_never(observation_choice)

    linear_system = MassSpringDamperSystem(
        number_of_connections=number_of_connections,
        damping_coefficient=damping_coefficient,
        observation_choice=observation_choice,
        observation_noise_covariance=observation_noise_covariance,
        process_noise_covariance=process_noise_variance
        * np.eye(2 * number_of_connections),
        number_of_systems=number_of_systems,
    )
    print(linear_system.observation)


if __name__ == "__main__":
    main()
