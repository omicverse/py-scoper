"""Spectral clustering core for pyscoper.

Faithful Python port of scoper's adaptive-threshold spectral clustering
(``passToClustering_lev2/3/4``, ``krnlMtxGenerator``, ``laplacianMtx``,
``makeAffinity``, ``infer``, ``findGapSmooth``, ``rangeAtoB``, ``likelihoods``).
"""
from __future__ import annotations

import numpy as np

from .density import gaussian_kde_r

__all__ = [
    "find_gap_smooth",
    "infer",
    "krnl_mtx_generator",
    "laplacian_mtx",
    "make_affinity",
    "range_a_to_b",
    "likelihoods",
    "pass_to_clustering",
]


def find_gap_smooth(vec):
    """Locate the valley between two density modes (port of ``findGapSmooth``)."""
    vec = np.asarray(vec, dtype=float)
    try:
        dens = gaussian_kde_r(vec)
    except Exception:
        return None
    y = dens["y"]
    x = dens["x"]
    # local minima: diff(sign(diff(y))) == 2
    s = np.sign(np.diff(y))
    idy = np.where(np.diff(s) == 2)[0] + 1
    if idy.size == 0:
        return None
    idx = idy[int(np.argmin(y[idy]))]
    return float(x[idx])


def infer(vec):
    """Infer an adaptive distance cut-off for a single row (port of ``infer``)."""
    vec = np.sort(np.asarray(vec, dtype=float))
    n = len(vec)
    d = np.nan
    if n > 2:
        g = find_gap_smooth(vec)
        if g is not None and not np.isnan(g):
            below = vec[vec <= g]
            d = g if (below.size and below.max() == 0) else (
                below.max() if below.size else g)
    if np.isnan(d):
        diff_vec = np.diff(vec)
        pos = diff_vec[diff_vec > 0]
        if pos.size and len(np.unique(pos)) == 1:
            d = np.ceil(np.mean(vec))
        else:
            x = int(np.argmax(diff_vec))
            if vec[x] == 0:
                gt0 = vec[vec > 0]
                nxt = gt0[0] if gt0.size else vec[x]
                d = np.mean([vec[x], nxt])
            else:
                d = vec[x]
    return float(d)


def krnl_mtx_generator(mtx):
    """Build the affinity kernel (port of ``krnlMtxGenerator``)."""
    mtx = np.asarray(mtx, dtype=float)
    n = mtx.shape[0]
    epsilon = np.array([infer(mtx[i]) for i in range(n)], dtype=float)
    krnl = np.ones((n, n), dtype=float)
    for i in range(n - 1):
        for j in range(i + 1, n):
            denom = epsilon[i] * epsilon[j]
            val = np.exp(-(mtx[i, j] ** 2) / denom) if denom != 0 else np.nan
            krnl[i, j] = krnl[j, i] = val
    krnl[np.isnan(krnl)] = 1.0
    return krnl


def laplacian_mtx(entry):
    """Unnormalized graph Laplacian ``L = D - W`` (port of ``laplacianMtx``)."""
    entry = np.asarray(entry, dtype=float)
    d = np.diag(entry.sum(axis=1))
    return d - entry


def make_affinity(mtx_o, mtx_k, thd):
    """Zero out kernel entries whose raw distance exceeds ``thd``."""
    mtx_k = np.array(mtx_k, dtype=float, copy=True)
    mtx_k[np.asarray(mtx_o, dtype=float) > thd] = 0.0
    return mtx_k


def range_a_to_b(x, a, b):
    """Linear rescale of ``x`` into ``[a, b]`` (port of ``rangeAtoB``)."""
    x = np.asarray(x, dtype=float)
    xmin, xmax = x.min(), x.max()
    if xmax == xmin:
        return np.zeros_like(x)
    return (b - a) * (x - xmin) / (xmax - xmin) + a


def likelihoods(tot_mtx, sh_mtx, mutab_mtx):
    """SHM-based likelihood matrix (port of ``likelihoods``)."""
    sh_mtx = np.asarray(sh_mtx, dtype=float)
    tot_mtx = np.asarray(tot_mtx, dtype=float)
    mutab_mtx = np.asarray(mutab_mtx, dtype=float)
    n = sh_mtx.shape[0]
    iu = np.triu_indices(n, k=1)
    sigma_sh = np.std(sh_mtx[iu], ddof=1) if iu[0].size > 1 else np.nan
    sigma_tot = np.std(tot_mtx[iu], ddof=1) if iu[0].size > 1 else np.nan
    if sigma_sh in (0,) or (isinstance(sigma_sh, float) and np.isnan(sigma_sh)) \
            or sigma_sh == 0:
        z = np.zeros((n, n))
    elif (isinstance(sigma_tot, float) and np.isnan(sigma_tot)) \
            or sigma_tot == 0:
        x = 1.0 - np.exp(-sh_mtx ** 2 / (2.0 * sigma_sh ** 2))
        z = mutab_mtx * x
    else:
        x = 1.0 - np.exp(-sh_mtx ** 2 / (2.0 * sigma_sh ** 2))
        y = np.exp(-(tot_mtx - sh_mtx) ** 2 / (2.0 * sigma_tot ** 2))
        z = mutab_mtx * x * y
    np.fill_diagonal(z, 1.0)
    return z


def _kmeans(x, k, iter_max=1000, nstart=1000, seed=0):
    """Lloyd k-means matching ``stats::kmeans`` defaults closely enough.

    Runs ``nstart`` random initialisations and keeps the lowest within-cluster
    sum of squares.  Returns a 1-based cluster label array.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    rng = np.random.RandomState(seed)
    best_labels = None
    best_wss = np.inf
    nstart = max(1, min(nstart, 200))  # cap for runtime; parity insensitive
    for _ in range(nstart):
        centers = x[rng.choice(n, size=k, replace=False)].copy()
        labels = np.zeros(n, dtype=int)
        for _it in range(iter_max):
            d = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
            new_labels = np.argmin(d, axis=1)
            if np.array_equal(new_labels, labels) and _it > 0:
                labels = new_labels
                break
            labels = new_labels
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centers[c] = x[mask].mean(axis=0)
        d = np.linalg.norm(x - centers[labels], axis=1)
        wss = float(np.sum(d ** 2))
        if wss < best_wss:
            best_wss = wss
            best_labels = labels.copy()
    return best_labels + 1


def _pass_lev4(aff_mtx, id_cluster):
    """Split spectral clusters into connected components (port of ``lev4``)."""
    aff_mtx = np.asarray(aff_mtx, dtype=float)
    n = aff_mtx.shape[0]
    new_id = np.full(n, -1, dtype=int)
    new_k = 0
    for cid in _unique_stable(id_cluster):
        members = np.where(np.asarray(id_cluster) == cid)[0]
        if members.size == 1:
            new_k += 1
            new_id[members[0]] = new_k
            continue
        sub = aff_mtx[np.ix_(members, members)]
        m = sub.shape[0]
        # connected components on the sub-affinity graph
        parent = list(range(m))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(m):
            for j in range(m):
                if sub[i, j] != 0:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)
        comp_label = {}
        for i in range(m):
            r = find(i)
            if r not in comp_label:
                new_k += 1
                comp_label[r] = new_k
            new_id[members[i]] = comp_label[r]
    return new_id


def _unique_stable(arr):
    seen = []
    for v in arr:
        if v not in seen:
            seen.append(v)
    return seen


def _pass_lev3(mtx, junc_length=None, threshold=None,
               iter_max=1000, nstart=1000, seed=0):
    """Spectral partitioning of a dissimilarity matrix (port of ``lev3``)."""
    mtx = np.asarray(mtx, dtype=float)
    n = mtx.shape[0]
    if n < 3:
        raise ValueError("SCOPer needs at least 3 unique data points")
    krnl = krnl_mtx_generator(mtx)
    if threshold is not None:
        aff = make_affinity(mtx, krnl, threshold * junc_length)
    else:
        nearest = []
        for col in range(n):
            gt0 = mtx[:, col][mtx[:, col] > 0]
            nearest.append(gt0.min() if gt0.size else np.nan)
        aff = make_affinity(mtx, krnl, np.nanmax(nearest))
    off = ~np.eye(n, dtype=bool)
    if np.all(aff[off] == 0):
        return {"id_cluster": np.arange(1, n + 1), "eigen_vals": np.zeros(n)}
    lap = laplacian_mtx(aff)
    vals, vecs = np.linalg.eigh(lap)  # ascending order, symmetric
    # R: eigen() returns descending; rev() -> ascending
    eigen_vals = range_a_to_b(vals, 0, 1)
    if np.all(eigen_vals == 0):
        return {"id_cluster": np.arange(1, n + 1), "eigen_vals": np.zeros(n)}
    dens = gaussian_kde_r(eigen_vals)
    peak_x = dens["x"][int(np.argmax(dens["y"]))]
    k_up = int(np.sum(eigen_vals < peak_x)) + 1
    if k_up in (1, 2) or k_up != k_up:
        k_up = n
    with np.errstate(divide="ignore", invalid="ignore"):
        log_diff = np.array([
            np.log10(eigen_vals[x]) - np.log10(eigen_vals[x - 1])
            for x in range(1, k_up)
        ])
    bad = ~np.isfinite(log_diff)
    nas = int(np.sum(bad))
    if nas == len(log_diff):
        k = nas + 1
    else:
        if nas > 0:
            rest = log_diff[nas:]
            k = nas + int(np.argmax(rest)) + 1 + 1
        else:
            k = nas + int(np.argmax(log_diff)) + 1 + 1
    if k == n:
        return {"id_cluster": np.arange(1, n + 1), "eigen_vals": np.zeros(n)}
    # eigenvectors: numpy ascending; R uses columns (n-k+1):n of descending
    # -> the k smallest eigenvalue eigenvectors
    eigen_vecs = vecs[:, :k]
    labels = _kmeans(np.round(eigen_vecs, 6), k,
                     iter_max=iter_max, nstart=nstart, seed=seed)
    labels = _pass_lev4(aff, labels)
    return {"id_cluster": labels, "eigen_vals": eigen_vals}


def pass_to_clustering(mtx, junc_length=None, threshold=None,
                       base_sim=0.95, iter_max=1000, nstart=1000, seed=0):
    """Top-level spectral partitioning of a dissimilarity matrix.

    Port of ``passToClustering_lev2``.  Returns a dict with ``id_cluster``
    (1-based labels) and ``eigen_vals``.
    """
    mtx = np.asarray(mtx, dtype=float)
    n = mtx.shape[0]
    bs = (1 - base_sim) * junc_length
    off = ~np.eye(n, dtype=bool)
    off_diags = np.unique(mtx[off]) if n > 1 else np.array([])
    if n == 1:
        return {"id_cluster": np.array([1]), "eigen_vals": np.array([0.0])}
    if off_diags.size == 1:
        if off_diags[0] > bs:
            return {"id_cluster": np.arange(1, n + 1),
                    "eigen_vals": np.zeros(n)}
        return {"id_cluster": np.ones(n, dtype=int),
                "eigen_vals": np.zeros(n)}
    if mtx.max() <= bs:
        return {"id_cluster": np.ones(n, dtype=int), "eigen_vals": np.zeros(n)}
    return _pass_lev3(mtx, junc_length=junc_length, threshold=threshold,
                      iter_max=iter_max, nstart=nstart, seed=seed)
