"""``ScoperClones`` result object and clonal-summary helpers.

Faithful port of scoper's ``ScoperClones`` S4 class plus
``calculateInterVsIntra`` and the effective-threshold computation embedded in
``defineClonesScoper``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .density import gaussian_kde_r

__all__ = ["ScoperClones", "calculate_inter_vs_intra", "effective_threshold"]


def calculate_inter_vs_intra(db: pd.DataFrame, clone: str,
                             vjl_gps: pd.DataFrame, junction: str = "junction",
                             cdr3: bool = False,
                             cdr3_col: Optional[str] = None) -> pd.DataFrame:
    """Port of ``calculateInterVsIntra``.

    For every V/J/length group, compute the maximum within-clone distance
    (``intra``) and the minimum between-clone distance (``inter``), each
    normalized by junction length.
    """
    from .distance import pairwise_dist_dna

    seq_col = cdr3_col if cdr3 else junction
    records = []
    for _, row in vjl_gps.iterrows():
        clones = str(row["clone_id"]).split(",")
        length = int(row["junction_length"])
        n_clones = len(clones)
        in_clones = db[clone].astype(str).isin(clones)
        sub = db[in_clones]
        # unique sequences keyed by clone label
        seqs = sub[seq_col].astype(str).tolist()
        names = sub[clone].astype(str).tolist()
        seen = {}
        for s, nm in zip(seqs, names):
            seen.setdefault((s, nm), None)
        u_seqs = [s for (s, nm) in seen]
        u_names = [nm for (s, nm) in seen]
        if len(u_seqs) == 0:
            continue
        dist = pairwise_dist_dna(u_seqs, iupac=False)
        names_arr = np.array(u_names)
        if n_clones == 1:
            sub_d = dist
            records.append((clones[0], "NA", "intra",
                            (sub_d.max() if sub_d.size else 0.0) / length))
        else:
            for i in range(n_clones - 1):
                mi = names_arr == clones[i]
                xx = dist[np.ix_(mi, mi)]
                records.append((clones[i], "NA", "intra",
                                (xx.max() if xx.size else 0.0) / length))
                for j in range(i + 1, n_clones):
                    mj = names_arr == clones[j]
                    xy = dist[np.ix_(mi, mj)]
                    records.append((clones[i], clones[j], "inter",
                                    (xy.min() if xy.size else 0.0) / length))
            last = clones[n_clones - 1]
            ml = names_arr == last
            yy = dist[np.ix_(ml, ml)]
            records.append((last, "NA", "intra",
                            (yy.max() if yy.size else 0.0) / length))
    return pd.DataFrame(records,
                        columns=["clone_id_x", "clone_id_y",
                                 "label", "distance"])


def effective_threshold(inter_intra: pd.DataFrame):
    """Port of the effective-threshold computation in ``defineClonesScoper``.

    Finds the crossing point between the intra- and inter-clone distance
    densities, restricted to a plausible region; returns ``nan`` if undefined.
    """
    if inter_intra is None or len(inter_intra) == 0:
        return float("nan")
    intra = inter_intra[(inter_intra["label"] == "intra")
                        & (inter_intra["distance"] > 0)]["distance"].to_numpy()
    inter = inter_intra[(inter_intra["label"] == "inter")
                        & (inter_intra["distance"] > 0)]["distance"].to_numpy()
    if len(intra) <= 5 or len(inter) <= 5:
        return float("nan")
    a, b = intra, inter
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    da = gaussian_kde_r(a, n=512)
    db_ = gaussian_kde_r(b, n=512)
    # evaluate both densities on a common grid restricted to [lo, hi]
    grid = np.linspace(lo, hi, 512)
    ya = np.interp(grid, da["x"], da["y"])
    yb = np.interp(grid, db_["x"], db_["y"])
    comp = (ya > yb).astype(int)
    cross = np.diff(comp)
    cross_x = grid[1:][cross != 0]
    if cross_x.size == 0:
        return float("nan")
    lo_b = a.mean() - a.std(ddof=1)
    hi_b = b.mean() + b.std(ddof=1)
    sel = cross_x[(lo_b < cross_x) & (cross_x < hi_b)]
    if sel.size == 0:
        return float("nan")
    return round(float(np.mean(sel)), 2)


class ScoperClones:
    """Result object holding clonal assignments and summary tables.

    Port of scoper's ``ScoperClones`` S4 class.  Returned by
    :func:`~pyscoper.clones.identicalClones` / ``hierarchicalClones`` /
    ``spectralClones`` when ``summarize_clones=True``.

    Attributes
    ----------
    db : pandas.DataFrame
        Input data with a ``clone_id`` column.
    vjl_groups : pandas.DataFrame
        Per V/J/length group summary statistics.
    inter_intra : pandas.DataFrame
        Inter- vs intra-clone distance table.
    eff_threshold : float
        Effective clonal-distance threshold (``nan`` if undefined).
    """

    def __init__(self, db, vjl_groups, inter_intra, eff_threshold):
        self.db = db
        self.vjl_groups = vjl_groups
        self.inter_intra = inter_intra
        self.eff_threshold = eff_threshold

    # --- S4-method equivalents ---
    def summary(self) -> pd.DataFrame:
        """Return the per-group summary table (S4 ``summary``)."""
        return self.vjl_groups

    def as_data_frame(self) -> pd.DataFrame:
        """Return the cloned data frame (S4 ``as.data.frame``)."""
        return pd.DataFrame(self.db)

    def to_dataframe(self) -> pd.DataFrame:
        """Alias for :meth:`as_data_frame`."""
        return self.as_data_frame()

    def plot(self, **kwargs):
        """Clonal-abundance / threshold plot (S4 ``plot``)."""
        from .plotting import plotCloneSummary
        return plotCloneSummary(self, **kwargs)

    def print(self) -> None:
        """Print the effective threshold (S4 ``print``)."""
        print(self.eff_threshold)

    def __repr__(self) -> str:
        n_clones = (self.db["clone_id"].nunique()
                    if "clone_id" in getattr(self.db, "columns", []) else 0)
        return (f"<ScoperClones: {len(self.db)} sequences, "
                f"{n_clones} clones, eff_threshold={self.eff_threshold}>")

    def __len__(self) -> int:
        return len(self.db)
