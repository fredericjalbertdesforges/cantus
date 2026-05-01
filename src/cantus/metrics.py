"""
Alignment-error metrics for benchmarking score followers.

Conventions:
    * Errors are computed in **performance frames** then converted to
      milliseconds using the feature frame rate.
    * Alignment Rate (AR) at threshold ``T`` is the fraction of perf
      frames whose absolute error is at most ``T``.
    * The summary helper returns a flat dict suitable for printing or
      logging into a benchmark table.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def alignment_error_frames(
    estimated: NDArray[np.int64],
    ground_truth: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Per-perf-frame absolute error in reference frames.

    The two arrays are truncated to their shared prefix length before
    comparison, so a slight length mismatch is tolerated silently. This
    is convenient when one tracker emits a few extra frames during
    end-of-sequence drainage.
    """
    n = min(len(estimated), len(ground_truth))
    diff: NDArray[np.int64] = np.abs(estimated[:n] - ground_truth[:n]).astype(np.int64)
    return diff


def alignment_rate(
    errors_ms: NDArray[np.floating],
    threshold_ms: float,
) -> float:
    """Fraction of perf frames whose error is within ``threshold_ms``."""
    if len(errors_ms) == 0:
        return 0.0
    return float(np.mean(errors_ms <= threshold_ms))


def summarise(errors_ms: NDArray[np.floating]) -> dict[str, float]:
    """Compact summary of an error distribution for benchmark tables."""
    if len(errors_ms) == 0:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
            "AR_50ms": 0.0,
            "AR_100ms": 0.0,
            "AR_250ms": 0.0,
        }
    return {
        "mean_ms": float(np.mean(errors_ms)),
        "median_ms": float(np.median(errors_ms)),
        "p95_ms": float(np.percentile(errors_ms, 95)),
        "p99_ms": float(np.percentile(errors_ms, 99)),
        "max_ms": float(np.max(errors_ms)),
        "AR_50ms": alignment_rate(errors_ms, 50.0),
        "AR_100ms": alignment_rate(errors_ms, 100.0),
        "AR_250ms": alignment_rate(errors_ms, 250.0),
    }
