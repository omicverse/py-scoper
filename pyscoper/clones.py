"""Clonal-partitioning entry points for pyscoper.

Faithful Python port of scoper's ``identicalClones``, ``hierarchicalClones``
and ``spectralClones`` (driven by ``defineClonesScoper``).

Each function partitions an AIRR-format :class:`pandas.DataFrame` into clonal
groups and returns it with an integer ``clone_id`` column.  With
``summarize_clones=True`` a :class:`~pyscoper.results.ScoperClones` object is
returned instead.
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .distance import (
    group_genes,
    pairwise_dist_aa,
    pairwise_dist_dna,
    pairwise_mut_matrix,
)
from .results import ScoperClones, calculate_inter_vs_intra, effective_threshold
from .spectral import likelihoods, pass_to_clustering

try:  # optional, only needed for hierarchical 'aa' and spectral 'vj'
    import pyalakazam as _al
except Exception:  # pragma: no cover
    _al = None
try:
    import pyshazam as _sh
except Exception:  # pragma: no cover
    _sh = None

__all__ = ["identicalClones", "hierarchicalClones", "spectralClones",
           "defineClonesScoper"]

_VALID_LOCI = {"IGH", "IGI", "IGK", "IGL", "TRA", "TRB", "TRD", "TRG"}
_HEAVY = {"IGH", "TRB", "TRD"}


# --------------------------------------------------------------------------
def _translate_dna(seqs: Sequence[str]) -> List[str]:
    """Translate nucleotide junctions to amino acids."""
    if _al is not None:
        return list(_al.translateDNA(list(seqs)))
    return [_simple_translate(s) for s in seqs]


_CODON = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L",
    "CTA": "L", "CTG": "L", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "TCT": "S", "TCC": "S",
    "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A",
    "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R",
    "CGA": "R", "CGG": "R", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def _simple_translate(seq: str) -> str:
    seq = str(seq).upper()
    out = []
    for i in range(0, len(seq) - 2, 3):
        out.append(_CODON.get(seq[i:i + 3], "X"))
    return "".join(out)


def _hclust_cutree(dist_mtx: np.ndarray, linkage: str, threshold: float):
    """Agglomerative clustering of a condensed distance matrix, cut at ``h``.

    Reproduces ``stats::hclust`` + ``cutree(h=threshold)`` for the three
    supported linkages.
    """
    from scipy.cluster.hierarchy import fcluster, linkage as sp_linkage
    from scipy.spatial.distance import squareform

    n = dist_mtx.shape[0]
    if n == 1:
        return np.array([1])
    method = {"single": "single", "complete": "complete",
              "average": "average"}[linkage]
    condensed = squareform(dist_mtx, checks=False)
    z = sp_linkage(condensed, method=method)
    # scipy fcluster with criterion 'distance' cuts the dendrogram at height h;
    # matches R cutree(h=...) (clusters merged at height <= h stay together).
    labels = fcluster(z, t=threshold, criterion="distance")
    return labels


# --- per-group helpers -----------------------------------------------------
def _identical_helper(seqs: Sequence[str], method: str) -> np.ndarray:
    """Port of ``identicalClones_helper``: clones = identical junctions."""
    if method == "aa":
        seqs = _translate_dna(seqs)
    # group_indices: 1-based, ordered by first occurrence
    seen = {}
    out = np.zeros(len(seqs), dtype=int)
    for i, s in enumerate(seqs):
        if s not in seen:
            seen[s] = len(seen) + 1
        out[i] = seen[s]
    return out


def _hierarchical_helper(seqs: Sequence[str], method: str, linkage: str,
                         normalize: str, threshold: float,
                         iupac: bool = False) -> np.ndarray:
    """Port of ``hierarchicalClones_helper``."""
    n = len(seqs)
    if method == "aa":
        work = _translate_dna(seqs)
    else:
        work = list(seqs)
    # unique sequences (first-occurrence order, like data.table by=seqs)
    uniq: dict = {}
    for i, s in enumerate(work):
        uniq.setdefault(s, []).append(i)
    seqs_unq = list(uniq.keys())
    ind_unq = list(uniq.values())
    n_unq = len(seqs_unq)
    if n_unq == 1:
        return np.ones(n, dtype=int)
    if method == "aa":
        dist_mtx = pairwise_dist_aa(seqs_unq)
    else:
        dist_mtx = pairwise_dist_dna(seqs_unq, iupac=iupac)
    if normalize == "len":
        junc_len = len(seqs_unq[0])
        dist_mtx = dist_mtx / junc_len
    labels_unq = _hclust_cutree(dist_mtx, linkage, threshold)
    out = np.zeros(n, dtype=int)
    for idx, members in enumerate(ind_unq):
        for m in members:
            out[m] = labels_unq[idx]
    return out


def _build_mut_matrices(germs_unq, seqs_unq, junc_length, cdr3, mutabs):
    """Port of ``pairwiseMutions`` (V-region SHM matrices)."""
    n = len(seqs_unq)
    lv = (_sh.IMGT_V.seqLength if _sh is not None else 312)
    if not cdr3:
        lv = lv - 3
    trim_l = junc_length

    def _trim(seq):
        x = list(str(seq))
        for p in range(lv, min(lv + trim_l, len(x))):
            x[p] = ""
        return "".join(x)

    seqs = [_trim(s) for s in seqs_unq]
    germs = [_trim(g) for g in germs_unq]
    # pad to equal length
    max_s = max(len(s) for s in seqs)
    seqs = [s.ljust(max_s, "N") for s in seqs]
    max_g = max(len(g) for g in germs)
    germs = [g.ljust(max_g, "N") for g in germs]

    if len(set(germs)) == 1:
        eff_germ = germs[0]
    else:
        eff_germ = _catch_all_consensus(list(dict.fromkeys(germs)))
    len_cons = min(len(seqs[0]), len(germs[0]), len(eff_germ))
    seqs = [s[:len_cons] for s in seqs]
    germs = [g[:len_cons] for g in germs]
    eff_germ = eff_germ[:len_cons]

    informative_pos = np.array(
        [sum(s.count(b) for b in "ACGT") for s in seqs], dtype=float)

    # mutation strings: effGerm + dist + seq char; "NA" for non-mutations / gaps
    mut = np.empty((n, len_cons), dtype=object)
    eff_chars = list(eff_germ)
    for i in range(n):
        si = seqs[i]
        for j in range(len_cons):
            a, b = eff_chars[j], si[j]
            d = _dna_match(a, b)
            tag = f"{a}{d}{b}"
            mut[i, j] = "NA" if ("0" in tag) else tag

    motif = np.zeros((n, len_cons), dtype=float)
    if mutabs is not None:
        for i in range(n):
            gi = germs[i]
            for j in range(2, len_cons - 2):
                m5 = gi[j - 2:j + 3]
                motif[i, j] = mutabs.get(m5, 0.0) if len(m5) == 5 else 0.0
    res = pairwise_mut_matrix(informative_pos, mut, motif)
    sh = res["sh_mtx"]
    tot = res["tot_mtx"]
    mu = res["mutab_mtx"]
    # symmetrize (R copies upper -> lower)
    for mat in (sh, tot, mu):
        il = np.tril_indices(n, k=-1)
        mat[il] = mat.T[il]
    return {"sh": sh, "tot": tot, "mutab": mu}


# IUPAC code for a set of nucleotides (shazam consensusSequence catchAll)
_NUC_TO_IUPAC = {
    frozenset("A"): "A", frozenset("C"): "C", frozenset("G"): "G",
    frozenset("T"): "T", frozenset("AC"): "M", frozenset("AG"): "R",
    frozenset("AT"): "W", frozenset("CG"): "S", frozenset("CT"): "Y",
    frozenset("GT"): "K", frozenset("ACG"): "V", frozenset("ACT"): "H",
    frozenset("AGT"): "D", frozenset("CGT"): "B", frozenset("ACGT"): "N",
}
_IUPAC_EXPAND = {
    "A": "A", "C": "C", "G": "G", "T": "T", "M": "AC", "R": "AG",
    "W": "AT", "S": "CG", "Y": "CT", "K": "GT", "V": "ACG", "H": "ACT",
    "D": "AGT", "B": "CGT", "N": "ACGT",
}


def _catch_all_consensus(sequences):
    """Port of shazam ``consensusSequence(method="catchAll")``.

    At each position, collapse all observed (non-gap) bases into the
    corresponding IUPAC ambiguity code.
    """
    if not sequences:
        return ""
    length = min(len(s) for s in sequences)
    out = []
    for j in range(length):
        bases = set()
        for s in sequences:
            ch = s[j].upper()
            if ch in _IUPAC_EXPAND:
                bases.update(_IUPAC_EXPAND[ch])
        if not bases:
            out.append("N")
        else:
            out.append(_NUC_TO_IUPAC.get(frozenset(bases), "N"))
    return "".join(out)


def _dna_match(a: str, b: str) -> str:
    """getDNAMatrix(gap=0) cell as a 1-char string for the mutation tag."""
    from .distance import _dna_pair_dist
    return "1" if _dna_pair_dist(a, b) >= 1.0 else "0"


def _spectral_helper(db_gp: pd.DataFrame, method: str, germline: str,
                     sequence: str, junction: str, threshold,
                     base_sim: float, mutabs, cdr3_col, cdr3: bool,
                     iter_max: int, nstart: int, seed: int) -> np.ndarray:
    """Port of ``spectralClones_helper`` (novj / vj modes)."""
    n = len(db_gp)
    seq_col = cdr3_col if cdr3 else junction

    if method == "vj":
        germs = db_gp[germline].astype(str).tolist()
        seqs = db_gp[sequence].astype(str).tolist()
        juncs = db_gp[seq_col].astype(str).tolist()
        junc_length = len(juncs[0])
        keys = [f"{seqs[i]}|{juncs[i]}|{germs[i]}" for i in range(n)]
        uniq: dict = {}
        for i, k in enumerate(keys):
            uniq.setdefault(k, []).append(i)
        n_unq = len(uniq)
        ind_unq = list(uniq.values())
        if n_unq == 1:
            return np.ones(n, dtype=int)
        u_keys = list(uniq.keys())
        seqs_unq = [k.split("|")[0] for k in u_keys]
        juncs_unq = [k.split("|")[1] for k in u_keys]
        germs_unq = [k.split("|")[2] for k in u_keys]
        dist_mtx = pairwise_dist_dna(juncs_unq, iupac=False)
        mut = _build_mut_matrices(germs_unq, seqs_unq, junc_length,
                                  cdr3, mutabs)
        lkl = likelihoods(mut["tot"], mut["sh"], mut["mutab"])
        disim = dist_mtx * (1.0 - lkl)
        if np.array_equal(disim, dist_mtx) or np.any(disim.sum(axis=1) == 0):
            # fall back to junction-only distance
            jseqs = db_gp[seq_col].astype(str).tolist()
            junc_length = len(jseqs[0])
            uniq = {}
            for i, s in enumerate(jseqs):
                uniq.setdefault(s, []).append(i)
            n_unq = len(uniq)
            ind_unq = list(uniq.values())
            if n_unq == 1:
                return np.ones(n, dtype=int)
            disim = pairwise_dist_dna(list(uniq.keys()), iupac=False)
    else:  # novj
        seqs = db_gp[seq_col].astype(str).tolist()
        junc_length = len(seqs[0])
        uniq = {}
        for i, s in enumerate(seqs):
            uniq.setdefault(s, []).append(i)
        n_unq = len(uniq)
        ind_unq = list(uniq.values())
        if n_unq == 1:
            return np.ones(n, dtype=int)
        disim = pairwise_dist_dna(list(uniq.keys()), iupac=False)

    res = pass_to_clustering(disim, junc_length=junc_length,
                             threshold=threshold, base_sim=base_sim,
                             iter_max=iter_max, nstart=nstart, seed=seed)
    labels_unq = np.asarray(res["id_cluster"])
    out = np.zeros(n, dtype=int)
    for idx, members in enumerate(ind_unq):
        for m in members:
            out[m] = labels_unq[idx]
    return out


# --------------------------------------------------------------------------
def defineClonesScoper(db: pd.DataFrame, model: str = "identical",
                       method: str = "nt", linkage: str = "single",
                       normalize: str = "len", germline: str = "germline_alignment",
                       sequence: str = "sequence_alignment", iupac: bool = False,
                       junction: str = "junction", v_call: str = "v_call",
                       j_call: str = "j_call", clone: str = "clone_id",
                       fields=None, locus: str = "locus", first: bool = False,
                       cdr3: bool = False, mod3: bool = False, max_n: int = 0,
                       threshold=None, base_sim: float = 0.95,
                       targeting_model=None, iter_max: int = 1000,
                       nstart: int = 1000, seed: int = 0,
                       verbose: bool = False,
                       summarize_clones: bool = False):
    """Core clonal-partitioning driver (port of ``defineClonesScoper``).

    Returns a partitioned DataFrame, or a dict with the summary tables when
    ``summarize_clones=True``.
    """
    if not isinstance(db, pd.DataFrame):
        raise TypeError("'db' must be a pandas DataFrame")
    db = db.copy()

    if model == "identical" and method not in ("nt", "aa"):
        raise ValueError("'method' should be 'nt' or 'aa' for model 'identical'")
    if model == "hierarchical":
        if normalize not in ("len", "none"):
            raise ValueError("'normalize' should be 'len' or 'none'")
        if linkage not in ("single", "average", "complete"):
            raise ValueError("'linkage' should be single/average/complete")
        if threshold is None or threshold > 1:
            raise ValueError("'threshold' should be a positive value < 1")
    if model == "spectral" and method not in ("novj", "vj"):
        raise ValueError("'method' should be 'novj' or 'vj' for model 'spectral'")

    temp_cols = ["vj_group", "vjl_group", "junction_l", "cdr3_col"]
    invalid = [clone] + temp_cols
    clash = [c for c in invalid if c in db.columns]
    if clash:
        raise ValueError(f"Column(s) {clash} already exist.")

    for col in [junction, v_call, j_call]:
        if col not in db.columns:
            raise ValueError(f"column '{col}' not found in db")

    # --- prepare_db ---
    db["junction_l"] = db[junction].astype(str).str.len()
    n_rmv_mod3 = 0
    if mod3:
        before = len(db)
        db = db[db["junction_l"] % 3 == 0].copy()
        n_rmv_mod3 = before - len(db)
    n_rmv_cdr3 = 0
    cdr3_col = None
    if cdr3:
        n_rmv_cdr3 = int((db["junction_l"] <= 6).sum())
        db = db[db["junction_l"] > 6].copy()
        db["cdr3_col"] = [s[3:l - 3] for s, l in
                          zip(db[junction].astype(str), db["junction_l"])]
        cdr3_col = "cdr3_col"
    n_rmv_N = 0
    if max_n is not None:
        import re
        cnt = db[junction].astype(str).apply(
            lambda s: len(re.findall(r"[^ATCG]", s)))
        n_rmv_N = int((cnt > max_n).sum())
        db = db[cnt <= max_n].copy()

    # V/J grouping
    if fields:
        groups = np.zeros(len(db), dtype=int)
        offset = 0
        for _, idx in db.groupby(list(fields)).groups.items():
            sub = db.loc[idx]
            g = group_genes(sub[v_call].tolist(), sub[j_call].tolist(),
                            first=first)
            groups[[db.index.get_loc(i) for i in idx]] = g + offset
            offset += g.max()
        db["vj_group"] = groups
    else:
        db["vj_group"] = group_genes(db[v_call].astype(str).tolist(),
                                     db[j_call].astype(str).tolist(),
                                     first=first)

    # vjl_group: distinct combination of vj_group, junction_l (+ fields)
    grp_cols = ["vj_group", "junction_l"] + (list(fields) if fields else [])
    vjl_codes = db.groupby(grp_cols, sort=False).ngroup()
    # re-number by first occurrence to mimic dplyr::group_indices order
    db["vjl_group"] = _first_occurrence_codes(vjl_codes.to_numpy())

    # --- heavy-chain filter (bulk mode) ---
    if locus not in db.columns:
        if _al is not None:
            db[locus] = _al.getLocus(db[v_call].tolist())
        else:  # pragma: no cover
            db[locus] = db[v_call].astype(str).str.extract(
                r"(IG[HKL]|TR[ABDG])")[0]
    db = db[db[locus].isin(_HEAVY)].copy()
    if len(db) == 0:
        raise ValueError("Cloning requires heavy chain sequences.")

    db = db.reset_index(drop=True)

    # --- per-group clustering ---
    vjl_ids = sorted(db["vjl_group"].unique())
    clone_strs = pd.Series(index=db.index, dtype=object)
    for gp_idx, vjl in enumerate(vjl_ids, start=1):
        sub = db[db["vjl_group"] == vjl]
        seqs = sub[cdr3_col if cdr3 else junction].astype(str).tolist()
        if model == "identical":
            labels = _identical_helper(seqs, method)
        elif model == "hierarchical":
            labels = _hierarchical_helper(seqs, method, linkage, normalize,
                                          threshold, iupac=iupac)
        else:  # spectral
            mutabs = None
            if targeting_model is not None:
                mutabs = getattr(targeting_model, "mutability",
                                 targeting_model)
            labels = _spectral_helper(sub, method, germline, sequence,
                                      junction, threshold, base_sim,
                                      mutabs, cdr3_col, cdr3,
                                      iter_max, nstart, seed)
        for pos, idx in enumerate(sub.index):
            clone_strs.loc[idx] = f"{gp_idx}_{labels[pos]}"

    # re-index clones globally (1-based, ordered)
    db[clone] = clone_strs.astype(str)
    uniq_clones = {}
    for c in db[clone]:
        if c not in uniq_clones:
            uniq_clones[c] = len(uniq_clones) + 1
    db[clone] = db[clone].map(uniq_clones)
    db = db.sort_values(clone, kind="stable").reset_index(drop=True)
    db[clone] = db[clone].astype(str)

    if summarize_clones:
        vjl_summary = _summarize_vjl(db, clone, v_call, j_call)
        inter_intra = calculate_inter_vs_intra(
            db, clone, vjl_summary, junction=junction, cdr3=cdr3,
            cdr3_col=cdr3_col)
        eff = effective_threshold(inter_intra)
        result = {"db": db.drop(columns=[c for c in temp_cols
                                         if c in db.columns]),
                  "vjl_gps": vjl_summary, "inter_intra": inter_intra,
                  "eff_threshold": eff}
        return result

    return db.drop(columns=[c for c in temp_cols if c in db.columns])


def _first_occurrence_codes(codes: np.ndarray) -> np.ndarray:
    mapping = {}
    out = np.zeros(len(codes), dtype=int)
    for i, c in enumerate(codes):
        if c not in mapping:
            mapping[c] = len(mapping) + 1
        out[i] = mapping[c]
    return out


def _summarize_vjl(db, clone, v_call, j_call):
    rows = []
    for vjl, sub in db.groupby("vjl_group", sort=True):
        rows.append({
            "vjl_group": vjl,
            "sequence_count": len(sub),
            "v_call": ",".join(sorted(set(
                ",".join(sub[v_call].astype(str)).split(",")))),
            "j_call": ",".join(sorted(set(
                ",".join(sub[j_call].astype(str)).split(",")))),
            "junction_length": int(sub["junction_l"].iloc[0]),
            "clone_count": sub[clone].nunique(),
            "clone_id": ",".join(map(str, sorted(set(sub[clone])))),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def identicalClones(db, method="nt", junction="junction", v_call="v_call",
                    j_call="j_call", clone="clone_id", fields=None,
                    locus="locus", first=False, cdr3=False, mod3=False,
                    max_n=0, verbose=False, summarize_clones=False):
    """Sequence-identity clonal partitioning (port of ``identicalClones``).

    Sequences are clones when they share the same V gene, J gene, junction
    length and an *identical* junction sequence.
    """
    res = defineClonesScoper(
        db, model="identical", method=method, junction=junction,
        v_call=v_call, j_call=j_call, clone=clone, fields=fields, locus=locus,
        first=first, cdr3=cdr3, mod3=mod3, max_n=max_n, verbose=verbose,
        summarize_clones=summarize_clones)
    if summarize_clones:
        return ScoperClones(res["db"], res["vjl_gps"], res["inter_intra"],
                            res["eff_threshold"])
    return res


def hierarchicalClones(db, threshold, method="nt", linkage="single",
                       normalize="len", iupac=False, junction="junction",
                       v_call="v_call", j_call="j_call", clone="clone_id",
                       fields=None, locus="locus", first=False, cdr3=False,
                       mod3=False, max_n=0, verbose=False,
                       summarize_clones=False):
    """Hierarchical-clustering clonal partitioning (port of ``hierarchicalClones``).

    Within each V/J/junction-length group, junctions are clustered by
    (length-normalized) nucleotide/AA Hamming distance and the dendrogram is
    cut at ``threshold``.
    """
    res = defineClonesScoper(
        db, model="hierarchical", method=method, linkage=linkage,
        normalize=normalize, iupac=iupac, threshold=threshold,
        junction=junction, v_call=v_call, j_call=j_call, clone=clone,
        fields=fields, locus=locus, first=first, cdr3=cdr3, mod3=mod3,
        max_n=max_n, verbose=verbose, summarize_clones=summarize_clones)
    if summarize_clones:
        return ScoperClones(res["db"], res["vjl_gps"], res["inter_intra"],
                            res["eff_threshold"])
    return res


def spectralClones(db, method="novj", germline="germline_alignment",
                   sequence="sequence_alignment", junction="junction",
                   v_call="v_call", j_call="j_call", clone="clone_id",
                   fields=None, locus="locus", targeting_model=None,
                   first=False, cdr3=False, mod3=False, max_n=0,
                   threshold=None, base_sim=0.95, iter_max=1000,
                   nstart=1000, seed=0, verbose=False,
                   summarize_clones=False):
    """Spectral-clustering clonal partitioning (port of ``spectralClones``).

    scoper's signature method.  ``method='novj'`` uses junction Hamming
    distance only; ``method='vj'`` additionally incorporates V-region SHM
    likelihoods.  Uses an adaptive-threshold spectral algorithm.
    """
    res = defineClonesScoper(
        db, model="spectral", method=method, germline=germline,
        sequence=sequence, junction=junction, v_call=v_call, j_call=j_call,
        clone=clone, fields=fields, locus=locus, targeting_model=targeting_model,
        first=first, cdr3=cdr3, mod3=mod3, max_n=max_n, threshold=threshold,
        base_sim=base_sim, iter_max=iter_max, nstart=nstart, seed=seed,
        verbose=verbose, summarize_clones=summarize_clones)
    if summarize_clones:
        return ScoperClones(res["db"], res["vjl_gps"], res["inter_intra"],
                            res["eff_threshold"])
    return res
