from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


@dataclass
class FormatConfig:
    draft: float
    publication: float

    def __call__(
        self,
        draft: bool = True,
    ) -> float:
        return self.draft if draft else self.publication


class Figure:
    def __init__(
        self,
        sup_xlabel: str = "",
        fig_size: tuple = (12, 8),
        fig_title: str | None = None,
        fig_layout: tuple[int, int] = (1, 1),
        column_ratio: tuple[float, ...] | None = None,
        row_ratio: tuple[float, ...] | None = None,
    ) -> None:
        self._sup_xlabel = sup_xlabel
        self._fig_size = fig_size
        self._fig_title = fig_title
        self._fig_layout = fig_layout

        self._fig = plt.figure(figsize=self._fig_size)
        self._grid_spec = gridspec.GridSpec(
            *self._fig_layout,
            figure=self._fig,
            width_ratios=column_ratio,
            height_ratios=row_ratio,
        )
        self._subplots: list[list[Axes]] = []
        for row in range(self._fig_layout[0]):
            self._subplots.append([])
            for col in range(self._fig_layout[1]):
                self._subplots[row].append(
                    self._fig.add_subplot(self._grid_spec[row, col])
                )

        self._default_color = "black"
        self._draft: bool = True
        self._font_size = FormatConfig(draft=12, publication=36)
        self._line_width = FormatConfig(draft=1, publication=5)
        self._marker_size = FormatConfig(draft=12, publication=500)

    @property
    def font_size(self) -> float:
        return self._font_size(self._draft)

    @property
    def line_width(self) -> float:
        return self._line_width(self._draft)

    @property
    def marker_size(self) -> float:
        return self._marker_size(self._draft)

    def format_config(self, draft: bool = True) -> Self:
        self._draft = draft
        return self

    @classmethod
    def remove_box(
        cls,
        ax: Axes,
    ) -> Axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        return ax

    @classmethod
    def add_horizontal_axis(
        cls,
        ax: Axes,
        position: (
            Literal["top", "center", "bottom", "zero", "one"]
            | tuple[Literal["axes", "data"], float]
        ),
        line_width: float,
    ) -> Axes:
        _position: tuple[Literal["axes", "data"], float]
        match position:
            case "top":
                _position = ("axes", 1.0)
            case "center":
                _position = ("axes", 0.5)
            case "bottom":
                _position = ("axes", 0.0)
            case "zero":
                _position = ("data", 0.0)
            case "one":
                _position = ("data", 1.0)
            case _:
                _position = position

        ax.xaxis.set_ticks_position("bottom")
        ax.spines["bottom"].set_position(_position)

        position_frame = "axes fraction" if _position[0] == "axes" else "data"
        position_value = _position[1]
        ax.annotate(
            "",
            xy=(1.05, position_value),
            xycoords=("axes fraction", position_frame),
            xytext=(-0.05, position_value),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=line_width,
                color="black",
                mutation_scale=30,
            ),
        )
        return ax

    @classmethod
    def add_vertical_axis(
        cls,
        ax: Axes,
        position: (
            Literal["left", "center", "right", "zero", "one"]
            | tuple[Literal["axes", "data"], float]
        ),
        line_width: float,
    ) -> Axes:
        _position: tuple[Literal["axes", "data"], float]
        match position:
            case "right":
                _position = ("axes", 1.0)
            case "center":
                _position = ("axes", 0.5)
            case "left":
                _position = ("axes", 0.0)
            case "zero":
                _position = ("data", 0.0)
            case "one":
                _position = ("data", 1.0)
            case _:
                _position = position

        ax.yaxis.set_ticks_position("left")
        ax.spines["left"].set_position(_position)

        position_frame = "axes fraction" if _position[0] == "axes" else "data"
        position_value = _position[1]
        ax.annotate(
            "",
            xy=(position_value, 1.05),
            xycoords=(position_frame, "axes fraction"),
            xytext=(position_value, -0.05),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=line_width,
                color="black",
                mutation_scale=30,
            ),
        )
        return ax

    @classmethod
    def set_arrow_axes(
        cls,
        ax: Axes,
        *,
        horizontal_axis_position: None
        | (
            Literal["top", "center", "bottom", "zero", "one"]
            | tuple[Literal["axes", "data"], float]
        ) = None,
        vertical_axis_position: None
        | (
            Literal["left", "center", "right", "zero", "one"]
            | tuple[Literal["axes", "data"], float]
        ) = None,
        line_width: float = 1,
        tick_label_font_size: float = 12,
    ) -> Axes:
        ax = cls.remove_box(ax)

        if horizontal_axis_position is not None:
            ax = cls.add_horizontal_axis(
                ax,
                horizontal_axis_position,
                line_width,
            )
            ax.tick_params(
                axis="x",
                which="major",
                width=line_width,
                length=3 * line_width,
                direction="out",
                labelsize=tick_label_font_size,
            )

        if vertical_axis_position is not None:
            ax = cls.add_vertical_axis(
                ax,
                vertical_axis_position,
                line_width,
            )
            ax.tick_params(
                axis="y",
                which="major",
                width=line_width,
                length=3 * line_width,
                direction="out",
                labelsize=tick_label_font_size,
            )

        return ax

    def plot(self) -> Self:
        if self._draft:
            if self._fig_title is not None:
                self._fig.suptitle(self._fig_title)
            if self._sup_xlabel != "":
                self._fig.supxlabel(self._sup_xlabel)
        self._fig.tight_layout()
        return self

    def save(self, filename: Path) -> None:
        if filename.suffix != ".pdf":
            filename = filename.with_suffix(".pdf")
        self._fig.savefig(
            filename,
            bbox_inches="tight",
            transparent=True,
        )


def show() -> None:
    plt.show()
