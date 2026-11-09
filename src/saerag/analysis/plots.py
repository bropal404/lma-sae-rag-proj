"""Shared figure style for the report: one palette, thin marks, recessive axes, value labels."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Categorical slots 1-3, checked for colour-blind separation. Slot 3 is under 3:1 contrast on
# white, so every bar carries a value label.
COLORS = {"ANSWER": "#2a78d6", "RETRIEVE": "#eb6834", "ABSTAIN": "#1baf7a",
          "closed": "#2a78d6", "bm25": "#eb6834", "gold": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def grouped_bars(ax, groups: list[str], series: dict[str, list[float]], *, pct=True,
                 errors: dict[str, list[tuple[float, float]]] | None = None) -> None:
    """Vertical grouped bars, one colour per series, value label on every bar."""
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, (name, vals) in enumerate(series.items()):
        pos = x - 0.4 + width * (i + 0.5)
        vals = np.asarray(vals, dtype=float)
        bars = ax.bar(pos, vals, width * 0.9, color=COLORS.get(name), label=name,
                      edgecolor="white", linewidth=0.8)
        tops = vals
        if errors and name in errors:
            lo, hi = np.array(errors[name], dtype=float).T
            ax.errorbar(pos, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor=MUTED,
                        elinewidth=0.8, capsize=2)
            tops = np.fmax(vals, hi)
        for b, v, top in zip(bars, vals, tops):
            if np.isfinite(v):
                ax.annotate(f"{v:.0%}" if pct else f"{v:.0f}", (b.get_x() + b.get_width() / 2, top),
                            xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                            fontsize=7, color=MUTED)
    ax.set_xticks(x, groups)
    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
