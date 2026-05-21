"""Bundled example data for pyscoper.

``ExampleDb`` is scoper's example AIRR-format database (heavy-chain BCR
sequences), exported verbatim from ``scoper::ExampleDb``.
"""
from __future__ import annotations

import os

import pandas as pd

__all__ = ["load_example_db", "ExampleDb"]

_DATA_DIR = os.path.join(os.path.dirname(__file__), "_data")


def load_example_db() -> pd.DataFrame:
    """Return scoper's ``ExampleDb`` as a :class:`pandas.DataFrame`.

    2000 AIRR-format BCR records with V/J calls, junctions and germline /
    sequence alignments.  The bundled copy retains scoper's original
    ``clone_id`` column; drop it before re-cloning.
    """
    path = os.path.join(_DATA_DIR, "ExampleDb.csv")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def __getattr__(name):  # lazy attribute
    if name == "ExampleDb":
        return load_example_db()
    raise AttributeError(name)
