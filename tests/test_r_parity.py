"""R-parity tests — pyscoper vs CRAN scoper 1.5.0.

The R driver (:file:`r_reference_driver.R`) runs scoper on its bundled
``ExampleDb`` and writes the resulting ``clone_id`` assignments.  pyscoper is
run on the *same* data and the clonal partitions are compared:

* ``identicalClones``    — deterministic -> Adjusted Rand Index == 1.0.
* ``hierarchicalClones`` — deterministic -> Adjusted Rand Index == 1.0.
* ``spectralClones``     — the spectral algorithm has stochastic k-means
  initialisation, so an exact match is not expected; we require ARI > 0.95.
* effective threshold    — exact agreement on ``ExampleDb`` (0.22).

Tests skip gracefully when the CMAP R env or scoper is unavailable.
"""
from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

import pandas as pd
import pytest

import pyscoper as sc

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
R_DRIVER = HERE / "r_reference_driver.R"
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"


def _adjusted_rand(labels_a, labels_b) -> float:
    """Adjusted Rand Index without a scikit-learn dependency."""
    a = pd.Categorical(pd.Series(labels_a).astype(str)).codes
    b = pd.Categorical(pd.Series(labels_b).astype(str)).codes
    n = len(a)
    ct = pd.crosstab(pd.Series(a), pd.Series(b)).to_numpy().astype(float)
    from math import comb
    sum_comb = sum(comb(int(v), 2) for v in ct.ravel())
    sum_a = sum(comb(int(v), 2) for v in ct.sum(axis=1))
    sum_b = sum(comb(int(v), 2) for v in ct.sum(axis=0))
    total = comb(n, 2)
    expected = (sum_a * sum_b) / total if total else 0.0
    maxv = 0.5 * (sum_a + sum_b)
    if maxv == expected:
        return 1.0
    return (sum_comb - expected) / (maxv - expected)


def _r_available() -> bool:
    if not R_DRIVER.exists():
        return False
    try:
        out = subprocess.run(
            ["bash", "-lc",
             f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
             "&& Rscript -e 'library(scoper); cat(\"OK\")'"],
            capture_output=True, text=True, timeout=180, check=False,
        )
        return out.returncode == 0 and "OK" in out.stdout
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _r_available(),
    reason="CMAP R env or scoper not installed.",
)


@pytest.fixture(scope="module")
def r_reference(tmp_path_factory):
    """Run the scoper R reference once; return the output directory."""
    out_dir = tmp_path_factory.mktemp("scoper_R")
    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
        f"&& Rscript {R_DRIVER} {out_dir}"
    )
    res = subprocess.run(
        ["bash", "-lc", cmd], capture_output=True, text=True, timeout=1800,
    )
    if res.returncode != 0:
        pytest.skip(f"R reference driver failed:\n{res.stderr[-2000:]}")
    return out_dir


@pytest.fixture(scope="module")
def db():
    return sc.load_example_db()


# ----------------------------------------------------------------------
def test_identical_clones_exact(r_reference, db):
    r = pd.read_csv(r_reference / "identical.csv",
                    dtype={"sequence_id": str}).set_index("sequence_id")
    py = sc.identicalClones(db).set_index("sequence_id")["clone_id"]
    common = r.index.intersection(py.index)
    assert len(common) == len(r) == len(py)
    ari = _adjusted_rand(r.loc[common, "clone_id"], py.loc[common])
    assert ari == pytest.approx(1.0), f"identicalClones ARI = {ari}"


def test_hierarchical_clones_exact(r_reference, db):
    r = pd.read_csv(r_reference / "hierarchical.csv",
                    dtype={"sequence_id": str}).set_index("sequence_id")
    py = sc.hierarchicalClones(db, threshold=0.15).set_index(
        "sequence_id")["clone_id"]
    common = r.index.intersection(py.index)
    ari = _adjusted_rand(r.loc[common, "clone_id"], py.loc[common])
    assert ari == pytest.approx(1.0), f"hierarchicalClones ARI = {ari}"


def test_spectral_clones_novj_vs_R(r_reference, db):
    r = pd.read_csv(r_reference / "spectral_novj.csv",
                    dtype={"sequence_id": str}).set_index("sequence_id")
    py = sc.spectralClones(db, method="novj", seed=1).set_index(
        "sequence_id")["clone_id"]
    common = r.index.intersection(py.index)
    ari = _adjusted_rand(r.loc[common, "clone_id"], py.loc[common])
    # spectral clustering has stochastic k-means -> exact match not expected
    assert ari > 0.95, f"spectralClones (novj) ARI = {ari:.4f} (expected > 0.95)"


def test_spectral_clones_vj_vs_R(r_reference, db):
    r = pd.read_csv(r_reference / "spectral_vj.csv",
                    dtype={"sequence_id": str}).set_index("sequence_id")
    py = sc.spectralClones(db, method="vj", seed=1).set_index(
        "sequence_id")["clone_id"]
    common = r.index.intersection(py.index)
    ari = _adjusted_rand(r.loc[common, "clone_id"], py.loc[common])
    assert ari > 0.95, f"spectralClones (vj) ARI = {ari:.4f} (expected > 0.95)"


def test_clone_count_close_to_R(r_reference, db):
    """Total clone count should be within a few percent of scoper's."""
    r = pd.read_csv(r_reference / "spectral_novj.csv")
    py = sc.spectralClones(db, method="novj", seed=1)
    r_n = r["clone_id"].nunique()
    py_n = py["clone_id"].nunique()
    rel = abs(r_n - py_n) / r_n
    assert rel < 0.05, f"clone count R={r_n} PY={py_n} (rel diff {rel:.3f})"


def test_effective_threshold_vs_R(r_reference, db):
    r = pd.read_csv(r_reference / "eff_threshold.csv")
    r_eff = float(r["eff_threshold"].iloc[0])
    res = sc.hierarchicalClones(db, threshold=0.15, summarize_clones=True)
    assert res.eff_threshold == pytest.approx(r_eff, abs=0.01), (
        f"eff_threshold R={r_eff} PY={res.eff_threshold}")
