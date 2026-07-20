from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle


def plot_simulation(
    times: Array,
    states: Array,
    controls: Array,
    running_costs: Array,
    save_path: Path,
) -> None:
    """Plot every batch member in a mass-spring-damper simulation."""
    number_of_connections = controls.shape[-1]
    figure, axes = plt.subplots(
        3, 1, figsize=(10, 9), sharex=True, constrained_layout=True
    )

    for batch_index in range(states.shape[1]):
        for connection_index in range(number_of_connections):
            label = f"Run {batch_index + 1}, mass {connection_index + 1}"
            axes[0].plot(
                times,
                states[:, batch_index, connection_index],
                linewidth=1.2,
                alpha=0.8,
                label=label,
            )
            axes[1].plot(
                times,
                controls[:, batch_index, connection_index],
                linewidth=1.2,
                alpha=0.8,
            )
        axes[2].plot(
            times,
            running_costs[:, batch_index],
            linewidth=1.2,
            alpha=0.8,
            label=f"Run {batch_index + 1}",
        )

    axes[0].set_title("Mass positions")
    axes[0].set_ylabel("Position (m)")
    axes[1].set_title("Control forces")
    axes[1].set_ylabel("Force (N)")
    axes[2].set_title("Running cost")
    axes[2].set_ylabel("Cost")
    axes[2].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(alpha=0.3)
    if states.shape[1] * number_of_connections <= 10:
        axes[0].legend(ncols=2, fontsize="small")
    if states.shape[1] <= 10:
        axes[2].legend(ncols=2, fontsize="small")

    figure.suptitle("LQG mass-spring-damper simulation")
    figure.savefig(save_path, dpi=160)
    plt.close(figure)


def render_animation(
    times: Array,
    states: Array,
    save_path: Path,
    fps: int = 30,
) -> None:
    """Render every batch member as a wall-connected horizontal chain."""
    number_of_connections = states.shape[-1] // 2
    equilibrium_positions = jnp.arange(1, number_of_connections + 1)
    positions = (
        states[:, :, :number_of_connections]
        + equilibrium_positions[None, None, :]
    )
    duration = times[-1] - times[0]
    frame_count = min(times.shape[0], max(2, round(float(duration) * fps)))
    frame_indices = jnp.linspace(0, times.shape[0] - 1, frame_count, dtype=int)

    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    margin = 0.5
    axis.set_xlim(
        min(0.0, float(jnp.min(positions))) - margin,
        float(jnp.max(positions)) + margin,
    )
    axis.set_ylim(-0.75, states.shape[1] - 0.25)
    axis.set_xlabel("Horizontal position (m)")
    axis.set_ylabel("Simulation batch")
    axis.set_yticks(jnp.arange(states.shape[1]))
    axis.set_yticklabels(
        [f"Run {index + 1}" for index in range(states.shape[1])]
    )
    axis.set_title("LQG mass-spring-damper simulation")
    axis.grid(alpha=0.25)

    colors = plt.colormaps["tab10"](
        jnp.linspace(0, 1, states.shape[1], endpoint=False)
    )
    mechanisms = []
    for batch_index, color in enumerate(colors):
        vertical_position = float(batch_index)
        axis.plot(
            [0, 0],
            [vertical_position - 0.18, vertical_position + 0.18],
            color="black",
            linewidth=4,
        )
        (spring,) = axis.plot([], [], color=color, linewidth=2, alpha=0.8)
        masses = []
        for _ in range(number_of_connections):
            mass = Circle(
                (0, vertical_position),
                0.09,
                facecolor=color,
                edgecolor="black",
                zorder=3,
            )
            axis.add_patch(mass)
            masses.append(mass)
        mechanisms.append((spring, masses))

    time_label = axis.text(0.02, 0.95, "", transform=axis.transAxes)

    def update(frame_index: int):
        state_index = frame_indices[frame_index]
        artists = []
        for batch_index, (spring, masses) in enumerate(mechanisms):
            mass_positions = positions[state_index, batch_index]
            spring.set_data(
                jnp.concatenate((jnp.zeros(1), mass_positions)),
                jnp.full(number_of_connections + 1, batch_index),
            )
            for mass, horizontal_position in zip(
                masses, mass_positions, strict=True
            ):
                mass.center = (horizontal_position, batch_index)
            artists.extend((spring, *masses))
        time_label.set_text(f"t = {times[state_index]:.2f} s")
        return *artists, time_label

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_count,
        interval=1000 / fps,
        blit=True,
    )
    writer = FFMpegWriter(
        fps=fps, metadata={"title": "LQG mass-spring-damper simulation"}
    )
    animation.save(save_path, writer=writer, dpi=140)
    plt.close(figure)
