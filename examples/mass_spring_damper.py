import click
import numpy as np

from system.linear import MassSpringDamperSystem


@click.command()
@click.option(
    "--observation-choice",
    type=click.Choice(
        ["last_position", "all_positions"],
        case_sensitive=False,
    ),
    default="last_position",
    help="Set the observation choice.",
)
def main(observation_choice: str) -> None:

    number_of_connections = 2
    number_of_systems = 3
    linear_system = MassSpringDamperSystem(
        number_of_connections=number_of_connections,
        damping_coefficient=0.0,
        observation_choice=observation_choice,
        observation_noise_covariance_R=np.diag([0.01]),
        # observation_choice="all_positions",
        # observation_noise_covariance_R=np.diag([0.01] * number_of_connections),
        process_noise_covariance_Q=0.1 * np.eye(2 * number_of_connections),
        number_of_systems=number_of_systems,
    )
    print(linear_system.observation)


if __name__ == "__main__":
    main()
