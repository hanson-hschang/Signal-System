import click
from tqdm import tqdm

from ss.system.markov import HiddenMarkovModel, MarkovChainCallback
from ss.utility import basic_config


@click.command()
@click.option(
    "--simulation-time-steps",
    type=click.IntRange(min=1),
    default=100,
    help="Set the simulation time steps (positive integers).",
)
@click.option(
    "--step-skip",
    type=click.IntRange(min=1),
    default=1,
    help="Set the step skip (positive integers).",
)
@click.option(
    "--number-of-systems",
    type=click.IntRange(min=1),
    default=1,
    help="Set the number of systems (positive integers).",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Set the verbose mode.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Set the debug mode.",
)
def main(
    simulation_time_steps: int,
    step_skip: int,
    number_of_systems: int,
    verbose: bool,
    debug: bool,
) -> None:
    result_directory = basic_config(__file__, verbose, debug)

    epsilon = 0.01
    transition_matrix = [
        [0, 0.5, 0.5],
        [epsilon, 1 - epsilon, 0],
        [1 - epsilon, 0, epsilon],
    ]
    emission_matrix = [
        [1, 0],
        [0, 1],
        [0, 1],
    ]

    markov_chain = HiddenMarkovModel(
        transition_matrix=transition_matrix,
        emission_matrix=emission_matrix,
        number_of_systems=number_of_systems,
    )
    system_callback = MarkovChainCallback(
        step_skip=step_skip,
        system=markov_chain,
    )

    current_time = 0.0
    for k in tqdm(range(simulation_time_steps)):
        system_callback.record(k, current_time)
        current_time = markov_chain.process(current_time)

    system_callback.record(simulation_time_steps, current_time)

    # Save the data
    system_callback.save(result_directory / "system.hdf5")


if __name__ == "__main__":
    main()
