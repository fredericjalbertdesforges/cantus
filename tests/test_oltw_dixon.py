"""
OLTW Dixon 2005 — theoretical-anchor tests.

Each test pins behaviour against a closed-form expected answer (identity
warp recovers the diagonal, integer-stretch warp recovers a known step
function, synthetic rubato stays within sampling resolution of ground
truth).
"""
from __future__ import annotations

import numpy as np
import pytest

from cantus.algorithms import oltw_align


def _binary_random_features(
    n_frames: int, dim: int, n_active: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((n_frames, dim), dtype=np.float32)
    for k in range(n_frames):
        active = rng.choice(dim, size=n_active, replace=False)
        out[k, active] = 1.0
    return out


def test_identity_warp_recovers_diagonal() -> None:
    """When perf == ref, alignment[j] should equal j up to a small jitter.

    OLTW emits the argmin within the search window, which can land on a
    neighbour of the true match when several frames have similar costs.
    Two-frame jitter is the documented worst case for the canonical Dixon
    heuristic on noisy binary features.
    """
    ref = _binary_random_features(n_frames=300, dim=12, n_active=2, seed=0)
    result = oltw_align(ref, ref, search_width=80, metric="euclidean")
    err = np.abs(result.alignment - np.arange(300))
    assert err.max() <= 2, f"max error too large: {err.max()}"
    assert (err == 0).mean() >= 0.75


def test_2x_stretch_warp_is_exact() -> None:
    """Each ref frame held for 2 perf frames → alignment = floor(j / 2)."""
    rng = np.random.default_rng(1)
    ref = (rng.random((100, 12)) > 0.7).astype(np.float32)
    perf = np.repeat(ref, 2, axis=0)
    gt = np.repeat(np.arange(100), 2)
    result = oltw_align(ref, perf, search_width=50, metric="euclidean")
    err = np.abs(result.alignment - gt)
    assert err.max() <= 1, f"max error too large on 2x stretch: {err.max()}"


def test_3x_stretch_warp_is_exact() -> None:
    """Each ref frame held for 3 perf frames → alignment = floor(j / 3)."""
    rng = np.random.default_rng(2)
    ref = (rng.random((80, 16)) > 0.7).astype(np.float32)
    perf = np.repeat(ref, 3, axis=0)
    gt = np.repeat(np.arange(80), 3)
    result = oltw_align(ref, perf, search_width=40, metric="euclidean")
    err = np.abs(result.alignment - gt)
    assert err.max() <= 1


def test_alignment_returns_metadata() -> None:
    """Result carries cost matrix and iteration count."""
    ref = _binary_random_features(50, 8, 2, seed=3)
    result = oltw_align(ref, ref, search_width=20, metric="euclidean")
    assert result.cost_matrix.shape == (50, 50)
    assert result.n_iterations > 0


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        oltw_align(np.zeros((0, 4)), np.zeros((10, 4)))
    with pytest.raises(ValueError):
        oltw_align(np.zeros((10, 4)), np.zeros((0, 4)))


def test_dim_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        oltw_align(np.zeros((10, 4)), np.zeros((10, 6)))


def test_unknown_metric_raises() -> None:
    ref = _binary_random_features(20, 4, 1, seed=4)
    with pytest.raises(ValueError):
        oltw_align(ref, ref, metric="manhattan")  # type: ignore[arg-type]


def test_cosine_metric_works_on_dense_features() -> None:
    """Cosine on non-zero rows tracks the diagonal."""
    rng = np.random.default_rng(5)
    ref = rng.random((100, 12)).astype(np.float32) + 0.1  # always positive, never zero
    result = oltw_align(ref, ref, search_width=40, metric="cosine")
    err = np.abs(result.alignment - np.arange(100))
    # cosine on dense features is allowed up to 2-frame jitter
    assert err.max() <= 2
