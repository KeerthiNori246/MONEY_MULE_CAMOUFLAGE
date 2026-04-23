# EGS — Embedding Geometry Signal module
# Phase 2 of the research pipeline (after SCI)
#
# Files:
#   train_gnn.py       — train GraphSAGE on full transaction graph
#   embed_snapshots.py — frozen inference → per-day .npy embedding matrices
#   (coming next)
#   compute_egs_signals.py  — prototype movement, dispersion, anisotropy
#   egs_timeseries.py       — iterate baselines → EGS time-series CSVs
#   egs_spike_detection.py  — spike analysis on EGS signals