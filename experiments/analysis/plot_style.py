"""Shared plot styling, figure dimensions, and export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from plotnine import ggsave

__all__ = [
    "COL_WIDTH",
    "PGF_RC_CONTEXT",
    "SCALE_FACTOR",
    "TEXT_WIDTH",
    "save_plot",
]

# Figure dimensions for two-column paper layout (in inches)
SCALE_FACTOR = 11 / 8.5  # Adjust for figure scaling if needed
TEXT_WIDTH = SCALE_FACTOR * 7.0319  # Two-column text width
COL_WIDTH = SCALE_FACTOR * 3.34  # Single-column text width

PGF_RC_CONTEXT = {
    "text.usetex": True,
    "pgf.rcfonts": False,  # Block Matplotlib from overriding metrics
    "font.family": "serif",
    "pgf.texsystem": "pdflatex",  # Use pdflatex for fast, predictable sizing
    "pgf.preamble": "\n".join(
        [
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[tt=false]{libertine}",
            r"\usepackage[libertine]{newtxmath}",
        ]
    ),
}


def save_plot(
    p: Any,
    out_path: Path,
    base_name: str,
    width: float,
    height: float,
    skip_pgf: bool = False,
) -> None:
    """Saves a plot (ggplot or composition) to PDF and conditionally to PGF."""
    pdf_path = out_path / f"{base_name}.pdf"
    if hasattr(p, "save_helper"):
        ggsave(p, str(pdf_path), width=width, height=height)
        if not skip_pgf:
            pgf_path = out_path / f"{base_name}.pgf"
            with mpl.rc_context(PGF_RC_CONTEXT):
                ggsave(p, str(pgf_path), width=width, height=height)
                print(f"Saved: {pdf_path} and {pgf_path}")
        else:
            print(f"Saved: {pdf_path}")
    else:
        fig = p.draw()
        fig.set_size_inches(width, height)
        fig.savefig(str(pdf_path))
        if not skip_pgf:
            pgf_path = out_path / f"{base_name}.pgf"
            with mpl.rc_context(PGF_RC_CONTEXT):
                fig.savefig(str(pgf_path))
                print(f"Saved: {pdf_path} and {pgf_path}")
        else:
            print(f"Saved: {pdf_path}")
        plt.close(fig)
