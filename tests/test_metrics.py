"""Tests for cantus.metrics."""
from __future__ import annotations

import numpy as np

from cantus.metrics import (
    alignment_error_frames,
    alignment_rate,
    summarise,
)


def test_alignment_error_frames_basic() -> None:
    estimated = np.array([0, 1, 3, 4], dtype=np.int64)
    truth = np.array([0, 1, 2, 4], dtype=np.int64)
    err = alignment_error_frames(estimated, truth)
    assert err.tolist() == [0, 0, 1, 0]


def test_alignment_error_frames_truncates_to_min_length() -> None:
    estimated = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    truth = np.array([0, 1, 2], dtype=np.int64)
    err = alignment_error_frames(estimated, truth)
    assert len(err) == 3


def test_alignment_rate_at_threshold() -> None:
    errors_ms = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    assert alignment_rate(errors_ms, 50.0) == 0.6  # ≤50ms: 0, 25, 50
    assert alignment_rate(errors_ms, 100.0) == 1.0
    assert alignment_rate(errors_ms, 0.0) == 0.2


def test_alignment_rate_empty() -> None:
    assert alignment_rate(np.array([]), 50.0) == 0.0


def test_summarise_keys() -> None:
    errors = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    summary = summarise(errors)
    expected_keys = {
        "mean_ms",
        "median_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
        "AR_50ms",
        "AR_100ms",
        "AR_250ms",
    }
    assert set(summary.keys()) == expected_keys


def test_summarise_values_for_known_distribution() -> None:
    errors = np.zeros(100)
    summary = summarise(errors)
    assert summary["mean_ms"] == 0.0
    assert summary["max_ms"] == 0.0
    assert summary["AR_50ms"] == 1.0


def test_summarise_empty() -> None:
    summary = summarise(np.array([]))
    assert all(v == 0.0 for v in summary.values())
