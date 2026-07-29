from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle

import jax.numpy as jnp
from jaxtyping import Array


def plot_simulation(
    times: Array,
    states: Array,
    save_path: Path | None = None,
) -> None:
    """Plot one cart-pole trajectory and optionally save the figure."""

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, constrained_layout=True)
    series = (
        (0, "Cart position", "Position (m)"),
        (2, "Pole angle", "Angle (rad)"),
        (1, "Cart velocity", "Velocity (m/s)"),
        (3, "Pole angular velocity", "Angular velocity (rad/s)"),
    )
    for axis, (state_index, title, ylabel) in zip(axes.flat, series, strict=True):
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

    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    figure.suptitle("Cart-pole simulation")

    if save_path is not None:
        figure.savefig(save_path, dpi=160)
    plt.close(figure)
    plt.close("all")


def render_animation(
    times: Array,
    states: Array,
    pole_length: float,
    save_path: Path,
    fps: int = 30,
) -> None:
    """Render one cart-pole trajectory as an MP4 animation."""

    duration = float(times[-1] - times[0])
    num_steps = int(times.shape[0])
    frame_count = min(num_steps, max(2, round(duration * fps)))
    frame_indices = jnp.linspace(0, num_steps - 1, frame_count, dtype=int)

    marker_radius = max(0.08, 0.06 * pole_length)
    x_positions = states[:, :, 0]
    horizontal_margin = pole_length + marker_radius

    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.set_xlim(
        float(x_positions.min()) - horizontal_margin,
        float(x_positions.max()) + horizontal_margin,
    )
    axis.set_ylim(-pole_length - marker_radius, pole_length + marker_radius)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Horizontal position (m)")
    axis.set_ylabel("Vertical position (m)")
    axis.set_title("Cart-pole simulation")
    axis.grid(alpha=0.25)
    axis.axhline(0, color="0.3", linewidth=2)

    colors = plt.colormaps["tab10"](jnp.linspace(0, 1, states.shape[1], endpoint=False))
    mechanisms = []
    for batch_index, color in enumerate(colors):
        cart = Circle(
            (0, 0),
            marker_radius,
            facecolor=color,
            edgecolor="black",
            alpha=0.8,
            zorder=4,
            label=f"Run {batch_index + 1}",
        )
        axis.add_patch(cart)
        (pole,) = axis.plot([], [], color=color, linewidth=4, zorder=5)
        bob = Circle((0, 0), marker_radius, color=color, zorder=6)
        axis.add_patch(bob)
        mechanisms.append((cart, pole, bob))

    # Commented out legend for large case.
    # if states.shape[1] > 1 and states.shape[1] <= 10:
    #     axis.legend(loc="lower right", fontsize="small")
    time_label = axis.text(0.02, 0.95, "", transform=axis.transAxes)

    def update(frame_index: int):  # type: ignore[no-untyped-def]
        state_index = frame_indices[frame_index]
        artists = []  # type: ignore[var-annotated]
        for batch_index, mechanism in enumerate(mechanisms):
            # Cart graphics: dumbell - rod connecting two circle
            cart, pole, bob = mechanism
            cart_position = states[state_index, batch_index, 0]
            pole_angle = states[state_index, batch_index, 2]
            pivot_x = cart_position
            pivot_y = 0
            bob_x = pivot_x + pole_length * jnp.sin(pole_angle)
            bob_y = pivot_y + pole_length * jnp.cos(pole_angle)

            cart.center = (cart_position, pivot_y)
            pole.set_data((pivot_x, bob_x), (pivot_y, bob_y))  # type: ignore[arg-type]
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
