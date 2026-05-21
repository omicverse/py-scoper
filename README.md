# py-scoper

**Pure-Python port of the R/CRAN package [`scoper`](https://scoper.readthedocs.io)**
— *Spectral Clustering for clOne Partitioning* — for identifying B-cell clones
from Adaptive Immune Receptor Repertoire sequencing (AIRR-Seq) data.

scoper is part of the [Immcantation](https://immcantation.readthedocs.io)
framework, developed by the Kleinstein Lab at Yale University. `py-scoper`
re-implements its three clonal-partitioning methods faithfully in
numpy / scipy / pandas / matplotlib — **no rpy2, no R runtime required**.

```python
import pyscoper as sc

db = sc.load_example_db()                      # scoper's ExampleDb (2000 BCRs)

ident = sc.identicalClones(db)                 # clones by identical junction
hier  = sc.hierarchicalClones(db, threshold=0.15)
spect = sc.spectralClones(db, method="novj")   # scoper's signature method

print(spect["clone_id"].nunique(), "clones")
```

Each function returns the input AIRR data frame with an integer `clone_id`
column.

## Installation

```bash
pip install pyscoper
```

Optional extras `pyalakazam` / `pyshazam` supply amino-acid translation,
germline-consensus and locus-inference helpers (`pip install pyscoper[full]`).
`py-scoper` ships internal fallbacks, so they are not strictly required.

## Methods

| Function | Description |
|----------|-------------|
| `identicalClones`    | Clones share the same V gene, J gene, junction length **and** an identical junction sequence. |
| `hierarchicalClones` | Junctions clustered by (length-normalized) nucleotide/AA Hamming distance within each V/J/length group; the dendrogram is cut at `threshold`. Linkages: `single`, `average`, `complete`. |
| `spectralClones`     | Adaptive-threshold spectral clustering — scoper's signature algorithm. `method="novj"` uses junction distance only; `method="vj"` additionally incorporates V-region somatic-hypermutation likelihoods. |

With `summarize_clones=True` a `ScoperClones` object is returned, carrying the
per-group summary table, the inter- vs intra-clone distance table, the
effective clonal-distance threshold, and a `plotCloneSummary` plot.

```python
res = sc.spectralClones(db, method="novj", summarize_clones=True)
res.summary()              # per V/J/length group statistics
res.eff_threshold          # effective distance threshold
sc.plotCloneSummary(res)   # clonal-distance distribution plot
```

## R parity

`py-scoper` is validated against **scoper 1.5.0** on its bundled `ExampleDb`
(2000 heavy-chain BCR sequences). Agreement is measured by the
Adjusted Rand Index (ARI) between the R and Python clonal partitions:

| Method | ARI vs scoper 1.5.0 | Notes |
|--------|--------------------:|-------|
| `identicalClones`    | **1.000** | deterministic — exact partition |
| `hierarchicalClones` | **1.000** | deterministic — exact partition |
| `spectralClones` (novj) | **~0.98** | spectral k-means is stochastic |
| `spectralClones` (vj)   | **~0.99** | spectral k-means is stochastic |
| effective threshold | exact (0.22) | |

The spectral method uses k-means with random initialisation, so an exact
partition is not expected; the parity tests require ARI > 0.95 and the total
clone count to be within a few percent of scoper's.

Run the parity suite (needs an R install with `scoper`):

```bash
pytest tests/ -q
```

## Public API

`identicalClones`, `hierarchicalClones`, `spectralClones`,
`defineClonesScoper`, `ScoperClones`, `plotCloneSummary`,
`calculate_inter_vs_intra`, `effective_threshold`, `fast_dist`,
`pairwise_dist_dna`, `pairwise_dist_aa`, `pairwise_mut_matrix`,
`count_invalid_bases`, `group_genes`, `pass_to_clustering`,
`krnl_mtx_generator`, `laplacian_mtx`, `make_affinity`, `likelihoods`,
`infer`, `find_gap_smooth`, `range_a_to_b`, `gaussian_kde_r`, `bw_nrd0`,
`load_example_db`.

## License

AGPL-3, matching the original scoper package. See `LICENSE`. Credit to the
Immcantation framework and the Kleinstein Lab (Yale University).

## References

- Nouri N, Kleinstein SH (2018). *A spectral clustering-based method for
  identifying clones from high-throughput B cell repertoire sequencing data.*
  Bioinformatics, i341–i349.
- Nouri N, Kleinstein SH (2020). *Somatic hypermutation analysis for improved
  identification of B cell clonal families from next-generation sequencing
  data.* PLOS Computational Biology, 16(6), e1007977.
- Gupta NT, et al. (2017). *Hierarchical clustering can identify B cell clones
  with high confidence in Ig repertoire sequencing data.* The Journal of
  Immunology, 2489–2499.
