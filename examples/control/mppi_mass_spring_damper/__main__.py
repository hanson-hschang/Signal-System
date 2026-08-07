from pathlib import Path

import click
import jax
import jax.numpy as jnp

from ss.control import MPPIController, QuadraticCost
from ss.system import (
    ControlChoice,
    MassSpringDamperSystem,
    ObservationChoice,
    simulate,
)

from .post_processing import plot_simulation, render_animation


@click.command()
@click.option("--number-of-connections", type=click.IntRange(min=1), default=2)
@click.option("--duration", type=click.FloatRange(min=0, min_open=True), default=2.0)
@click.option("--time-step", type=click.FloatRange(min=0, min_open=True), default=0.02)
@click.option("--horizon", type=click.IntRange(min=1), default=50)
@click.option("--num-rollouts", type=click.IntRange(min=1), default=512)
@click.option("--batch-size", type=click.IntRange(min=1), default=1)
@click.option(
    "--temperature",
    type=click.FloatRange(min=0, min_open=True),
    default=1.0,
)
@click.option(
    "--noise-sigma",
    type=click.FloatRange(min=0, min_open=True),
    default=1.0,
)
@click.option(
    "--control-limit",
    type=click.FloatRange(min=0, min_open=True),
    default=10.0,
)
@click.option("--seed", type=int, default=0)
@click.option("--save-dir", type=click.Path(file_okay=False, path_type=Path))
def main(
    number_of_connections: int,
    duration: float,
    time_step: float,
    horizon: int,
    num_rollouts: int,
    batch_size: int,
    temperature: float,
    noise_sigma: float,
    control_limit: float,
    seed: int,
    save_dir: Path | None,
) -> None:
    """Regulate a fully observed mass-spring-damper chain with MPPI."""
    num_steps = round(duration / time_step)
    state_dim = 2 * number_of_connections
    system = MassSpringDamperSystem(
        number_of_connections=number_of_connections,
        time_step=time_step,
        observation_choice=ObservationChoice.ALL_STATES,
        control_choice=ControlChoice.ALL_FORCES,
        process_noise_covariance=0.01 * jnp.eye(state_dim),
        observation_noise_covariance=0.0 * jnp.eye(state_dim),
        initial_state_standard_deviation=1.0,
        batch_size=batch_size,
    )
    cost = QuadraticCost(
        state_weight=jnp.eye(system.state_dim),
        control_weight=0.1 * jnp.eye(system.control_dim),
        terminal_scale=10.0,
    )
    controller = MPPIController(
        control_dim=system.control_dim,
        batch_size=system.batch_size,
        rollout_system=system,
        running_cost=cost.running_cost,
        terminal_cost=cost.terminal_cost,
        horizon=horizon,
        num_rollouts=num_rollouts,
        temperature=temperature,
        noise_sigma=noise_sigma,
        control_limit=control_limit,
    )

    random_key = jax.random.PRNGKey(seed)
    initial_key, simulation_key = jax.random.split(random_key)
    initial_state = system.initial_state(initial_key)
    result = simulate(
        system,
        0.0,
        num_steps,
        initial_state,
        simulation_key,
        controller=controller,
    )
    running_costs = cost.running_cost(result.states[:-1], result.controls)

    click.echo(f"final_state={result.states[-1]}")
    click.echo(f"total_running_cost={jnp.sum(running_costs, axis=0) * system.time_step}")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_simulation(
            result.times[:-1],
            result.states[:-1],
            result.controls,
            running_costs,
            save_dir / "mppi_mass_spring_damper_plot.png",
        )
        render_animation(
            result.times,
            result.states,
            save_dir / "mppi_mass_spring_damper.mp4",
        )
        click.echo(f"Saved MPPI results to {save_dir}")


if __name__ == "__main__":
    main()
