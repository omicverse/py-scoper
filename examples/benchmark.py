"""Benchmark pyscoper's three clonal-partitioning methods on ExampleDb.

Reports wall-clock time and clone counts for identicalClones,
hierarchicalClones and spectralClones (novj / vj).

Usage:  python examples/benchmark.py
"""
from __future__ import annotations

import time
import warnings

import pyscoper as sc

warnings.filterwarnings("ignore")


def _timeit(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    n_clones = out["clone_id"].nunique()
    print(f"  {label:<28s} {dt:7.2f}s   {n_clones:5d} clones   "
          f"{len(out):5d} sequences")
    return dt


def main():
    db = sc.load_example_db()
    print(f"ExampleDb: {len(db)} sequences, {db.shape[1]} columns\n")
    print("Method                          time      clones    sequences")
    print("-" * 64)
    _timeit("identicalClones (nt)",
            lambda: sc.identicalClones(db))
    _timeit("identicalClones (aa)",
            lambda: sc.identicalClones(db, method="aa"))
    _timeit("hierarchicalClones (single)",
            lambda: sc.hierarchicalClones(db, threshold=0.15))
    _timeit("hierarchicalClones (average)",
            lambda: sc.hierarchicalClones(db, threshold=0.15,
                                          linkage="average"))
    _timeit("spectralClones (novj)",
            lambda: sc.spectralClones(db, method="novj", seed=1))
    _timeit("spectralClones (vj)",
            lambda: sc.spectralClones(db, method="vj", seed=1))
    print("-" * 64)

    # summary object
    res = sc.spectralClones(db, method="novj", seed=1, summarize_clones=True)
    print(f"\nScoperClones summary: {len(res.summary())} V/J/length groups, "
          f"effective threshold = {res.eff_threshold}")


if __name__ == "__main__":
    main()
