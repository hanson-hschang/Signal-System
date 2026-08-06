from pathlib import Path

import click
import jax
import jax.numpy as jnp

from ss.system import (
    MassSpringDamperSystem,
    ObservationChoice,
    ControlChoice,
    simulate,
)

from .post_processing import plot_simulation, render_animation


@click.command()
@click.option("--number-of-connections", type=click.IntRange(min=1), default=2)
@click.option("--duration", type=click.FloatRange(min=0, min_open=True), default=5.0)
@click.option("--time-step", type=click.FloatRange(min=0, min_open=True), default=0.02)
@click.option(
    "--damping-coefficient",
    type=click.FloatRange(min=0),
    default=0.1,
)
@click.option("--batch-size", type=click.IntRange(min=1), default=1)
@click.option("--seed", type=int, default=0)
@click.option("--save-dir", type=click.Path(file_okay=False, path_type=Path))
def main(
    number_of_connections: int,
    duration: float,
    time_step: float,
    damping_coefficient: float,
    batch_size: int,
    seed: int,
    save_dir: Path | None,
) -> None:
    """Simulate an uncontrolled mass-spring-damper chain."""
    num_steps = round(duration / time_step)
    state_dim = 2 * number_of_connections
    system = MassSpringDamperSystem(
        number_of_connections=number_of_connections,
        damping_coefficient=damping_coefficient,
        time_step=time_step,
        observation_choice=ObservationChoice.ALL_POSITIONS,
        control_choice=ControlChoice.NO_CONTROL,
        process_noise_covariance=0.01 * jnp.eye(state_dim),
        observation_noise_covariance=0.01 * jnp.eye(number_of_connections),
        initial_state_standard_deviation=1.0,
        batch_size=batch_size,
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
    )

    click.echo(f"final_state={result.states[-1]}")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_path = save_dir / "mass_spring_damper_plot.png"
        animation_path = save_dir / "mass_spring_damper_animation.mp4"
        plot_simulation(result.times, result.states, save_path=plot_path)
        render_animation(result.times, result.states, animation_path)
        click.echo(f"Saved plot to {plot_path}")
        click.echo(f"Saved animation to {animation_path}")


if __name__ == "__main__":
    main()
