from typing import List, Optional, Self, Tuple

from dataclasses import dataclass
from pathlib import Path

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
        fig_size: Tuple = (12, 8),
        fig_title: Optional[str] = None,
        fig_layout: Tuple[int, int] = (1, 1),
    ) -> None:
        self._sup_xlabel = sup_xlabel
        self._fig_size = fig_size
        self._fig_title = fig_title
        self._fig_layout = fig_layout

        self._fig = plt.figure(figsize=self._fig_size)
        self._grid_spec = gridspec.GridSpec(
            *self._fig_layout, figure=self._fig
        )
        self._subplots: List[List[Axes]] = []
        for row in range(self._fig_layout[0]):
            self._subplots.append([])
            for col in range(self._fig_layout[1]):
                self._subplots[row].append(
                    self._fig.add_subplot(self._grid_spec[row, col])
                )

        self._draft: bool = True
        self._font_size = FormatConfig(draft=12, publication=36)
        self._line_width = FormatConfig(draft=1, publication=5)

    @property
    def font_size(self) -> float:
        return self._font_size(self._draft)

    @property
    def line_width(self) -> float:
        return self._line_width(self._draft)

    def format_config(self, draft: bool = True) -> Self:
        self._draft = draft
        return self

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
