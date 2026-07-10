"""Analyst report generator — the last mile of the attribution doctrine.

Turns a recording's calibrated outputs (CSV + JSON artifacts) into ONE
self-contained HTML report an analyst can read: data quality, per-clip channel
timelines, direction-aware node table, flagged-window drill-down, optional
coupling lane. Pure pandas/numpy/stdlib — no torch, no GPU, no network access
(the production box is air-gapped: every byte is inline).
"""
