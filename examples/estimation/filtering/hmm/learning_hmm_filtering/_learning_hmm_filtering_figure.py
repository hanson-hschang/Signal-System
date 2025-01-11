from matplotlib.axes import Axes


def add_optimal_loss_line(
    ax: Axes,
    optimal_loss: float,
    arrowhead_x_offset_ratio: float = 0.05,
    text_offset: tuple[float, float] = (64, 32),
    text_coordinates: str = "offset pixels",
) -> None:
    ax.axhline(y=optimal_loss, color="black", linestyle="--")
    bbox = dict(boxstyle="round", fc="0.8")
    arrowprops = dict(
        arrowstyle="->",
        connectionstyle="angle,angleA=0,angleB=90,rad=10",
    )
    xlim_min, xlim_max = ax.get_xlim()
    xlim_range = xlim_max - xlim_min
    ax.annotate(
        f"optimal loss: {optimal_loss:.2f}\n(based on HMM-filter)",
        (xlim_min + arrowhead_x_offset_ratio * xlim_range, optimal_loss),
        xytext=text_offset,
        textcoords=text_coordinates,
        bbox=bbox,
        arrowprops=arrowprops,
    )
