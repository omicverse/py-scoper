"""pyscoper — pure-Python port of the R/CRAN package **scoper**.

scoper (Spectral Clustering for clOne Partitioning) identifies B-cell clones
from Adaptive Immune Receptor Repertoire sequencing (AIRR-Seq) data and is
part of the Immcantation framework (Kleinstein lab, Yale).

This package is a faithful Python re-implementation — numpy / scipy / pandas /
matplotlib only, no rpy2 — providing the three clonal-partitioning methods:

* :func:`identicalClones`    — clones by *identical* junction sequence.
* :func:`hierarchicalClones` — hierarchical clustering of junction distances.
* :func:`spectralClones`     — the adaptive-threshold spectral-clustering
  method (``novj`` / ``vj`` modes), scoper's signature algorithm.

Each returns the input AIRR data frame with an integer ``clone_id`` column;
with ``summarize_clones=True`` a :class:`ScoperClones` object carrying the
summary tables and the :func:`plotCloneSummary` plot is returned instead.

Original R package: https://scoper.readthedocs.io  (AGPL-3, Immcantation /
Kleinstein lab).  References: Nouri & Kleinstein 2018 *Bioinformatics*;
Nouri & Kleinstein 2020 *PLOS Comput Biol*; Gupta et al. 2017 *J Immunol*.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .clones import (
    defineClonesScoper,
    hierarchicalClones,
    identicalClones,
    spectralClones,
)
from .data import load_example_db
from .density import bw_nrd0, gaussian_kde_r
from .distance import (
    count_invalid_bases,
    fast_dist,
    group_genes,
    pairwise_dist_aa,
    pairwise_dist_dna,
    pairwise_mut_matrix,
)
from .plotting import plotCloneSummary
from .results import ScoperClones, calculate_inter_vs_intra, effective_threshold
from .spectral import (
    find_gap_smooth,
    infer,
    krnl_mtx_generator,
    laplacian_mtx,
    likelihoods,
    make_affinity,
    pass_to_clustering,
    range_a_to_b,
)

__all__ = [
    "__version__",
    # main clonal-partitioning entry points
    "identicalClones",
    "hierarchicalClones",
    "spectralClones",
    "defineClonesScoper",
    # result object + summary helpers
    "ScoperClones",
    "calculate_inter_vs_intra",
    "effective_threshold",
    "plotCloneSummary",
    # distance / grouping primitives
    "fast_dist",
    "pairwise_dist_dna",
    "pairwise_dist_aa",
    "pairwise_mut_matrix",
    "count_invalid_bases",
    "group_genes",
    # spectral-clustering internals
    "pass_to_clustering",
    "krnl_mtx_generator",
    "laplacian_mtx",
    "make_affinity",
    "likelihoods",
    "infer",
    "find_gap_smooth",
    "range_a_to_b",
    # density / bandwidth (R density() port)
    "gaussian_kde_r",
    "bw_nrd0",
    # data
    "load_example_db",
]
