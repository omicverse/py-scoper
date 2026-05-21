"""Smoke tests for pyscoper — exercise every public entry point."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import pyscoper as sc

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
def test_all_exports_importable():
    for name in sc.__all__:
        assert hasattr(sc, name), f"missing export: {name}"


def test_example_db_loads():
    db = sc.load_example_db()
    assert isinstance(db, pd.DataFrame)
    assert len(db) == 2000
    for col in ("junction", "v_call", "j_call", "germline_alignment",
                "sequence_alignment"):
        assert col in db.columns


# --- distance primitives ---------------------------------------------------
def test_fast_dist_basic():
    d = sc.fast_dist(["ATGC", "ATGC", "TTGC"])
    assert d.shape == (3, 3)
    assert d[0, 0] == 0
    assert d[0, 2] == 1
    assert np.array_equal(d, d.T)


def test_fast_dist_n_matches_any():
    d = sc.fast_dist(["ATGC", "NTGC"])
    assert d[0, 1] == 0  # N matches any base


def test_count_invalid_bases():
    assert sc.count_invalid_bases(["ATGC", "ATGC"]) == 0
    assert sc.count_invalid_bases(["ATGZ", "ATGC"]) == 1


def test_pairwise_dist_aa():
    d = sc.pairwise_dist_aa(["MKL", "MKL", "MRL"])
    assert d[0, 1] == 0
    assert d[0, 2] == 1


def test_group_genes():
    v = ["Homsap IGHV3-11*05 F", "Homsap IGHV3-11*01 F",
         "Homsap IGHV1-2*02 F"]
    j = ["Homsap IGHJ4*02 F", "Homsap IGHJ4*02 F", "Homsap IGHJ6*01 F"]
    g = sc.group_genes(v, j)
    assert g[0] == g[1]  # same gene, different alleles -> same group
    assert g[0] != g[2]


def test_pairwise_mut_matrix():
    mut = np.array([["A1G", "NA"], ["A1G", "NA"]], dtype=object)
    motif = np.zeros((2, 2))
    res = sc.pairwise_mut_matrix([1.0, 1.0], mut, motif)
    assert set(res) == {"sh_mtx", "tot_mtx", "mutab_mtx"}


# --- spectral internals ----------------------------------------------------
def test_density_port():
    d = sc.gaussian_kde_r(np.array([0.0, 1, 1, 2, 5, 8, 9, 10]))
    assert len(d["x"]) == 512
    assert np.all(d["y"] >= 0)


def test_infer_and_gap():
    vec = [0, 0, 1, 1, 5, 6, 6, 7]
    d = sc.infer(vec)
    assert np.isfinite(d)


def test_laplacian_and_kernel():
    mtx = np.array([[0, 1, 5], [1, 0, 5], [5, 5, 0]], dtype=float)
    k = sc.krnl_mtx_generator(mtx)
    assert k.shape == (3, 3)
    lap = sc.laplacian_mtx(k)
    assert np.allclose(lap.sum(axis=1), 0)


def test_range_a_to_b():
    out = sc.range_a_to_b([0, 5, 10], 0, 1)
    assert np.isclose(out[0], 0) and np.isclose(out[-1], 1)


# --- clonal partitioning ---------------------------------------------------
@pytest.fixture(scope="module")
def db():
    return sc.load_example_db()


def test_identical_clones(db):
    out = sc.identicalClones(db)
    assert "clone_id" in out.columns
    assert len(out) == 2000
    assert out["clone_id"].nunique() > 100


def test_hierarchical_clones(db):
    out = sc.hierarchicalClones(db, threshold=0.15)
    assert "clone_id" in out.columns
    assert out["clone_id"].nunique() > 100


def test_hierarchical_aa(db):
    out = sc.hierarchicalClones(db.iloc[:200], threshold=0.15, method="aa")
    assert "clone_id" in out.columns


def test_spectral_clones_novj(db):
    out = sc.spectralClones(db, method="novj", seed=1)
    assert "clone_id" in out.columns
    assert out["clone_id"].nunique() > 100


def test_spectral_clones_vj(db):
    out = sc.spectralClones(db.iloc[:300], method="vj", seed=1)
    assert "clone_id" in out.columns


def test_summarize_clones_object(db):
    res = sc.hierarchicalClones(db, threshold=0.15, summarize_clones=True)
    assert isinstance(res, sc.ScoperClones)
    assert isinstance(res.summary(), pd.DataFrame)
    assert isinstance(res.as_data_frame(), pd.DataFrame)
    assert "clone_id" in res.as_data_frame().columns
    assert len(res) == 2000


def test_plot_clone_summary(db):
    res = sc.hierarchicalClones(db, threshold=0.15, summarize_clones=True)
    ax = sc.plotCloneSummary(res, silent=True)
    assert ax is not None


def test_identical_aa_method(db):
    out = sc.identicalClones(db, method="aa")
    assert out["clone_id"].nunique() > 50


def test_clone_id_clash_raises(db):
    bad = db.copy()
    bad["clone_id"] = "x"
    with pytest.raises(ValueError):
        sc.identicalClones(bad)
