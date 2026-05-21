#!/usr/bin/env Rscript
# R reference driver for py-scoper parity tests.
#
# Runs scoper 1.5.0's identicalClones, hierarchicalClones and spectralClones
# on the bundled ExampleDb and writes the resulting `clone_id` assignments
# (keyed by sequence_id) so the Python port can be compared against them.
#
# Usage:  Rscript r_reference_driver.R <out_dir>

suppressMessages({
  library(scoper)
})

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else "."
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

data(ExampleDb, package = "scoper")
db <- ExampleDb

# --- identicalClones (deterministic) --------------------------------------
ic <- identicalClones(db, verbose = FALSE)
write.csv(data.frame(sequence_id = ic$sequence_id,
                     clone_id = ic$clone_id),
          file.path(out_dir, "identical.csv"), row.names = FALSE)

# --- hierarchicalClones (deterministic) -----------------------------------
hc <- hierarchicalClones(db, threshold = 0.15, verbose = FALSE)
write.csv(data.frame(sequence_id = hc$sequence_id,
                     clone_id = hc$clone_id),
          file.path(out_dir, "hierarchical.csv"), row.names = FALSE)

# --- spectralClones, novj mode (has stochastic k-means) -------------------
set.seed(1)
sc_novj <- spectralClones(db, method = "novj", verbose = FALSE)
write.csv(data.frame(sequence_id = sc_novj$sequence_id,
                     clone_id = sc_novj$clone_id),
          file.path(out_dir, "spectral_novj.csv"), row.names = FALSE)

# --- spectralClones, vj mode ----------------------------------------------
set.seed(1)
sc_vj <- spectralClones(db, method = "vj", verbose = FALSE)
write.csv(data.frame(sequence_id = sc_vj$sequence_id,
                     clone_id = sc_vj$clone_id),
          file.path(out_dir, "spectral_vj.csv"), row.names = FALSE)

# --- effective threshold from a summarized run ----------------------------
hc_sum <- hierarchicalClones(db, threshold = 0.15, summarize_clones = TRUE)
write.csv(data.frame(eff_threshold = hc_sum@eff_threshold),
          file.path(out_dir, "eff_threshold.csv"), row.names = FALSE)

cat("R reference driver complete.\n")
