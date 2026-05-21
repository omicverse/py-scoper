"""Distance utilities for pyscoper.

Faithful Python ports of the scoper C++ helpers (``fastDist.cpp``,
``RcppMutation.cpp``) and the V/J gene grouping logic (``alakazam::groupGenes``).

The Hamming-distance routines reproduce ``getDNAMatrix(gap = 0)`` semantics:
``A,C,G,T`` mismatches cost 1, ``N`` matches any base (cost 0), ``?`` matches
only ``?``.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

__all__ = [
    "fast_dist",
    "pairwise_dist_dna",
    "pairwise_dist_aa",
    "count_invalid_bases",
    "pairwise_mut_matrix",
    "group_genes",
]

# nucleotide encoding identical to scoper's fastDist.cpp ``code_char``
_CODE = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, "?": 5}


def _encode(seqs: Sequence[str]):
    n = len(seqs)
    if n == 0:
        raise ValueError("empty input")
    length = len(seqs[0])
    enc = np.empty((n, length), dtype=np.int16)
    for i, s in enumerate(seqs):
        if len(s) != length:
            raise ValueError("All sequences must have the same length")
        for p, ch in enumerate(s):
            c = _CODE.get(ch.upper(), 255)
            if c == 255:
                raise ValueError("Only A,C,G,T,N,? are allowed")
            enc[i, p] = c
    return enc, n, length


def fast_dist(seqs: Sequence[str]) -> np.ndarray:
    """Pairwise Hamming distance matrix (port of ``fastDist_rcpp``).

    ``N`` matches any of A/C/G/T (and N); ``?`` matches only ``?``.
    Returns an ``n x n`` integer matrix with a zero diagonal.
    """
    enc, n, length = _encode(seqs)
    match = np.zeros((n, n), dtype=np.int64)
    for p in range(length):
        col = enc[:, p]
        # known bases A,C,G,T -> identical-symbol matches
        for code in (0, 1, 2, 3):
            idx = np.where(col == code)[0]
            if idx.size:
                match[np.ix_(idx, idx)] += 1
        ns = np.where(col == 4)[0]
        known = np.where(col < 4)[0]
        if ns.size:
            if known.size:
                match[np.ix_(ns, known)] += 1
                match[np.ix_(known, ns)] += 1
            match[np.ix_(ns, ns)] += 1  # N-N is a match
        q = np.where(col == 5)[0]
        if q.size:
            match[np.ix_(q, q)] += 1  # ? matches only ?
    dist = length - match
    np.fill_diagonal(dist, 0)
    return dist.astype(np.int64)


def count_invalid_bases(seqs: Iterable[str]) -> int:
    """Number of sequences containing characters outside A/C/G/T/N/?.

    Port of ``countSeqsWithInvalidBases_rcpp``.  ``None``/NaN counts as bad.
    """
    bad = 0
    for s in seqs:
        if s is None or (isinstance(s, float) and np.isnan(s)):
            bad += 1
            continue
        if any(_CODE.get(ch.upper(), 255) == 255 for ch in str(s)):
            bad += 1
    return bad


# --- IUPAC-aware distance via a substitution matrix --------------------------
# getDNAMatrix(gap=0) semantics: ambiguous codes match if their nucleotide
# sets overlap; N and the gap characters (-, .) match *everything* (cost 0);
# ? matches only ?.
_IUPAC_DNA = {
    "A": {"A"}, "C": {"C"}, "G": {"G"}, "T": {"T"},
    "M": {"A", "C"}, "R": {"A", "G"}, "W": {"A", "T"},
    "S": {"C", "G"}, "Y": {"C", "T"}, "K": {"G", "T"},
    "V": {"A", "C", "G"}, "H": {"A", "C", "T"},
    "D": {"A", "G", "T"}, "B": {"C", "G", "T"},
}
# characters whose distance to any other (non-?) character is 0
_DNA_WILD = {"N", "-", "."}


def _dna_pair_dist(a: str, b: str) -> float:
    """getDNAMatrix(gap=0) cell value for a pair of (possibly IUPAC) chars."""
    a = a.upper()
    b = b.upper()
    if a == b:
        return 0.0
    if a == "?" or b == "?":
        return 1.0
    if a in _DNA_WILD or b in _DNA_WILD:
        return 0.0
    sa = _IUPAC_DNA.get(a)
    sb = _IUPAC_DNA.get(b)
    if sa is None or sb is None:
        return 1.0
    return 0.0 if sa & sb else 1.0


def pairwise_dist_dna(seqs: Sequence[str], iupac: bool = False) -> np.ndarray:
    """Pairwise Hamming distance for nucleotide sequences.

    With ``iupac=False`` uses the fast A/C/G/T/N/? routine; with ``iupac=True``
    falls back to the full IUPAC substitution matrix (``getDNAMatrix(gap=0)``).
    """
    if not iupac:
        try:
            return fast_dist(seqs).astype(float)
        except ValueError:
            pass
    n = len(seqs)
    length = len(seqs[0])
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = 0.0
            si, sj = seqs[i], seqs[j]
            for p in range(length):
                d += _dna_pair_dist(si[p], sj[p])
            dist[i, j] = dist[j, i] = d
    return dist


# amino-acid distance: hamming over translated sequence with gap=0 semantics
def pairwise_dist_aa(seqs: Sequence[str]) -> np.ndarray:
    """Pairwise Hamming distance for amino-acid sequences (``getAAMatrix(gap=0)``)."""
    n = len(seqs)
    length = len(seqs[0])
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        si = seqs[i]
        for j in range(i + 1, n):
            sj = seqs[j]
            d = sum(1 for p in range(length) if si[p] != sj[p]
                    and si[p] not in ("X", ".", "-")
                    and sj[p] not in ("X", ".", "-"))
            dist[i, j] = dist[j, i] = float(d)
    return dist


def pairwise_mut_matrix(informative_pos, mut_mtx, motif_mtx):
    """Port of ``pairwiseMutMatrixRcpp``.

    Parameters
    ----------
    informative_pos : array, length n
    mut_mtx : object array (n x m) of per-position mutation strings, ``"NA"`` for
        non-informative positions.
    motif_mtx : float array (n x m) of motif mutabilities.

    Returns
    -------
    dict with ``sh_mtx``, ``tot_mtx``, ``mutab_mtx`` (upper-triangular filled).
    """
    informative_pos = np.asarray(informative_pos, dtype=float)
    mut_mtx = np.asarray(mut_mtx, dtype=object)
    motif_mtx = np.asarray(motif_mtx, dtype=float)
    n, m = mut_mtx.shape
    sh = np.zeros((n, n))
    tot = np.zeros((n, n))
    mutab = np.ones((n, n))
    for i in range(n - 1):
        ri = mut_mtx[i]
        for j in range(i + 1, n):
            rj = mut_mtx[j]
            shmut = 0.0
            allmut = 0.0
            nonmutab = 1.0
            for k in range(m):
                a, b = ri[k], rj[k]
                if a == b and a != "NA" and b != "NA":
                    shmut += 1
                    nonmutab *= 1.0 - (motif_mtx[i, k] + motif_mtx[j, k]) / 2.0
                if a != "NA":
                    allmut += 1
                if b != "NA":
                    allmut += 1
            nu = 2.0 / (informative_pos[i] + informative_pos[j])
            sh[i, j] = 2.0 * nu * shmut
            tot[i, j] = nu * allmut
            mutab[i, j] = nonmutab
    return {"sh_mtx": sh, "tot_mtx": tot, "mutab_mtx": mutab}


# --- V/J gene grouping (alakazam::groupGenes) --------------------------------
def _get_gene(allele: str) -> str:
    """Reduce an IMGT allele string to the gene level (``alakazam::getGene``).

    ``"Homsap IGHV3-11*05 F"`` -> ``"IGHV3-11"``.  Drops a leading species
    token and any whitespace-delimited functionality suffix, then strips the
    allele (``*NN``) component.
    """
    s = str(allele).strip()
    # take the IG/TR token if present (handles "Homsap IGHV3-11*05 F")
    for tok in s.split():
        if tok[:2] in ("IG", "TR"):
            s = tok
            break
    else:
        s = s.split()[0] if s.split() else s
    return s.split("*")[0]


def _expand_vjl(v: str, j: str, first: bool = False,
                sep_anno: str = ",", sep_chain: str = ";") -> List[str]:
    """Port of ``alakazam::getAllVJL`` (junction-length-free, bulk case).

    Returns the list of ``V@J`` gene combinations covered by a row.
    """
    multi_chain = sep_chain in v
    if multi_chain:
        vs = v.split(sep_chain)
        js = j.split(sep_chain)
    else:
        vs = [v]
        js = [j]
    v_lists = []
    for chunk in vs:
        alleles = chunk.split(sep_anno)
        if first:
            alleles = alleles[:1]
        genes = []
        for a in alleles:
            g = _get_gene(a)
            if g not in genes:
                genes.append(g)
        v_lists.append(genes)
    j_lists = []
    for chunk in js:
        alleles = chunk.split(sep_anno)
        if first:
            alleles = alleles[:1]
        genes = []
        for a in alleles:
            g = _get_gene(a)
            if g not in genes:
                genes.append(g)
        j_lists.append(genes)
    combos: List[str] = []
    for vg_list, jg_list in zip(v_lists, j_lists):
        for jg in jg_list:
            for vg in vg_list:
                combo = f"{vg}@{jg}"
                if combo not in combos:
                    combos.append(combo)
    return combos


def group_genes(v_calls: Sequence[str], j_calls: Sequence[str],
                first: bool = False) -> np.ndarray:
    """Assign a V/J group index to every row.

    Faithful port of ``alakazam::groupGenes`` for the bulk (no ``junc_len``,
    no single-cell) case used by scoper: two sequences are placed in the same
    group if their sets of ``V@J`` gene combinations are linked through a
    connected component (shared combos chain groups together).

    Returns a 1-based integer array of group ids, one per input row.
    """
    n = len(v_calls)
    row_combos = [_expand_vjl(str(v_calls[i]), str(j_calls[i]), first=first)
                  for i in range(n)]
    # union-find over distinct V@J combos
    combo_id: dict = {}
    parent: List[int] = []

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for combos in row_combos:
        ids = []
        for c in combos:
            if c not in combo_id:
                combo_id[c] = len(parent)
                parent.append(len(parent))
            ids.append(combo_id[c])
        for k in range(1, len(ids)):
            union(ids[0], ids[k])

    # component id per combo, then per row (first-seen order -> 1-based)
    comp_label: dict = {}
    group = np.zeros(n, dtype=np.int64)
    for i, combos in enumerate(row_combos):
        root = find(combo_id[combos[0]])
        if root not in comp_label:
            comp_label[root] = len(comp_label) + 1
        group[i] = comp_label[root]
    return group
