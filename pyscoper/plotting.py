"""Plotting for pyscoper.

Port of scoper's ``plotCloneSummary`` — the clonal-distance (inter- vs
intra-clone) histogram/density with the effective-threshold marker.
"""
from __future__ import annotations

import numpy as np

from .density import gaussian_kde_r

__all__ = ["plotCloneSummary"]


def plotCloneSummary(data, xmin=None, xmax=None, binwidth=None,
                     title=None, size=0.75, silent=False, ax=None):
    """Plot the inter- vs intra-clone distance distribution.

    Parameters
    ----------
    data : ScoperClones
        Result of a clonal-partitioning call with ``summarize_clones=True``.
    xmin, xmax : float, optional
        x-axis limits.
    binwidth : float, optional
        Histogram bin width.
    title : str, optional
        Plot title.  Defaults to the effective threshold.
    size : float
        Density-line width.
    silent : bool
        If ``True`` return the axes without drawing.
    ax : matplotlib axes, optional
        Existing axes to draw onto.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from .results import ScoperClones

    if not isinstance(data, ScoperClones):
        raise TypeError(
            "Input data is not a ScoperClones object. Use the output of "
            "identicalClones/hierarchicalClones/spectralClones with "
            "summarize_clones=True.")
    import matplotlib.pyplot as plt

    eff = data.eff_threshold
    xdf = data.inter_intra
    intra = xdf[(xdf["label"] == "intra")
                & (xdf["distance"] > 0)]["distance"].to_numpy()
    inter = xdf[(xdf["label"] == "inter")
                & (xdf["distance"] > 0)]["distance"].to_numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    allv = np.concatenate([intra, inter]) if (len(intra) + len(inter)) else \
        np.array([0.0, 1.0])
    if binwidth is None:
        rng = allv.max() - allv.min()
        binwidth = rng / 40 if rng > 0 else 0.025
    bins = np.arange(allv.min(), allv.max() + binwidth, binwidth)

    if len(intra) > 0:
        ax.hist(intra, bins=bins, density=True, color="0.3",
                edgecolor="white", alpha=0.85,
                label="maximum-distance within clones")
        d = gaussian_kde_r(intra)
        ax.plot(d["x"], d["y"], color="0.3", lw=size * 2)
    if len(inter) > 0:
        ax.hist(inter, bins=bins, density=True, color="0.6",
                edgecolor="white", alpha=0.75,
                label="minimum-distance between clones")
        d = gaussian_kde_r(inter)
        ax.plot(d["x"], d["y"], color="0.6", lw=size * 2)
    else:
        import warnings
        warnings.warn("No inter clonal distance detected.")

    if eff is not None and not (isinstance(eff, float) and np.isnan(eff)):
        ax.axvline(eff, color="0.3", ls="--", lw=size * 2)
        plot_title = f"Effective threshold= {eff}"
    else:
        plot_title = "Effective threshold not found"

    ax.set_xlabel("Normalized hamming distance")
    ax.set_ylabel("Density")
    ax.set_title(title if title is not None else plot_title)
    if xmin is not None or xmax is not None:
        ax.set_xlim(xmin, xmax)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    if not silent:
        import matplotlib.pyplot as plt
        plt.tight_layout()
    return ax
