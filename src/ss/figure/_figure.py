from typing import List, Optional, Self, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


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

    def plot(self) -> Self:
        if self._fig_title is not None:
            self._fig.suptitle(self._fig_title)
        if self._sup_xlabel != "":
            self._fig.supxlabel(self._sup_xlabel)
        self._fig.tight_layout()
        return self


def show() -> None:
    plt.show()
