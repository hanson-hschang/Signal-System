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
    """Plot the state, control, and cost of an MPPI simulation."""
    figure, axes = plt.subplots(
        3, 2, figsize=(12, 10), sharex=True, constrained_layout=True
    )
    series = (
        (0, "Cart position", "Position (m)"),
        (2, "Pole angle", "Angle (rad)"),
        (1, "Cart velocity", "Velocity (m/s)"),
        (3, "Pole angular velocity", "Angular velocity (rad/s)"),
    )
    for axis, (state_index, title, ylabel) in zip(
        axes[:2].flat, series, strict=True
    ):
        for batch_index in range(states.shape[1]):
            axis.plot(
                times,
                states[:, batch_index, state_index],
                linewidth=1.3,
                alpha=0.8,
                label=f"Run {batch_index + 1}",
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)

    if states.shape[1] > 1 and states.shape[1] <= 10:
        axes[0, 0].legend(ncols=2, fontsize="small")

    for batch_index in range(controls.shape[1]):
        axes[2, 0].plot(
            times,
            controls[:, batch_index, 0],
            linewidth=1.3,
            alpha=0.8,
        )
    axes[2, 0].set_title("MPPI control")
    axes[2, 0].set_ylabel("Force (N)")
    axes[2, 0].grid(alpha=0.3)

    for batch_index in range(running_costs.shape[1]):
        axes[2, 1].plot(
            times,
            running_costs[:, batch_index],
            linewidth=1.3,
            alpha=0.8,
        )
    axes[2, 1].set_title("Running cost")
    axes[2, 1].set_ylabel("Cost")
    axes[2, 1].grid(alpha=0.3)

    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    figure.suptitle("MPPI cart-pole simulation")
    figure.savefig(save_path, dpi=160)
    plt.close(figure)


def render_animation(
    times: Array,
    states: Array,
    pole_length: float,
    save_path: Path,
    fps: int = 30,
) -> None:
    """Render cart-pole trajectories as an MP4 animation."""
    duration = times[-1] - times[0]
    num_steps = times.shape[0]
    frame_count = min(num_steps, max(2, round(duration * fps)))
    frame_indices = jnp.linspace(0, num_steps - 1, frame_count, dtype=int)

    marker_radius = max(0.08, 0.06 * pole_length)
    x_positions = states[:, :, 0]
    horizontal_margin = pole_length + marker_radius

    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.set_xlim(
        x_positions.min() - horizontal_margin,
        x_positions.max() + horizontal_margin,
    )
    axis.set_ylim(-pole_length - marker_radius, pole_length + marker_radius)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Horizontal position (m)")
    axis.set_ylabel("Vertical position (m)")
    axis.set_title("MPPI cart-pole simulation")
    axis.grid(alpha=0.25)
    axis.axhline(0, color="0.3", linewidth=2)

    colors = plt.colormaps["tab10"](
        jnp.linspace(0, 1, states.shape[1], endpoint=False)
    )
    mechanisms = []
    for color in colors:
        cart = Circle(
            (0, 0),
            marker_radius,
            facecolor=color,
            edgecolor="black",
            alpha=0.8,
            zorder=4,
        )
        axis.add_patch(cart)
        (pole,) = axis.plot([], [], color=color, linewidth=4, zorder=5)
        bob = Circle((0, 0), marker_radius, color=color, zorder=6)
        axis.add_patch(bob)
        mechanisms.append((cart, pole, bob))

    time_label = axis.text(0.02, 0.95, "", transform=axis.transAxes)

    def update(frame_index: int):
        state_index = frame_indices[frame_index]
        artists = []
        for batch_index, mechanism in enumerate(mechanisms):
            cart, pole, bob = mechanism
            cart_position = states[state_index, batch_index, 0]
            pole_angle = states[state_index, batch_index, 2]
            bob_x = cart_position + pole_length * jnp.sin(pole_angle)
            bob_y = pole_length * jnp.cos(pole_angle)

            cart.center = (cart_position, 0)
            pole.set_data((cart_position, bob_x), (0, bob_y))
            bob.center = (bob_x, bob_y)
            artists.extend(mechanism)

        time_label.set_text(f"t = {times[state_index]:.2f} s")
        return *artists, time_label

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_count,
        interval=1000 / fps,
        blit=True,
    )
    writer = FFMpegWriter(fps=fps, metadata={"title": "Cart-pole simulation"})
    animation.save(save_path, writer=writer, dpi=140)
    plt.close(figure)
