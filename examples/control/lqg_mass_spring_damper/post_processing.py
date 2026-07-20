from pathlib import Path

import matplotlib.pyplot as plt
from jaxtyping import Array


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
