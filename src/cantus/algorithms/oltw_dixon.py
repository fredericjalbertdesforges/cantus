"""
Online Time Warping, Dixon 2005.

Reference:
    Dixon, S. (2005). An On-Line Time Warping Algorithm for Tracking Musical
    Performances. IJCAI 2005.

Cantus follows the original paper as written: ``MAX_RUN_COUNT = 3``,
three directions (``ROW``, ``COLUMN``, ``BOTH``), no warmup skip.
These choices differ from ``pymatchmaker`` (Park 2025); see the
project ``CHANGELOG.md`` for rationale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

ROW = "R"
COLUMN = "C"
BOTH = "B"

Metric = Literal["cosine", "euclidean"]


@dataclass
class OLTWResult:
    """Result of an OLTW alignment.

    Attributes:
        alignment: ``alignment[j]`` is the estimated reference frame for
            performance frame ``j``.
        cost_matrix: full cumulative cost matrix of shape ``[M, N]``.
            Useful for diagnostic plots; safe to discard.
        n_iterations: number of decision steps taken by the OLTW loop.
    """

    alignment: NDArray[np.int64]
    cost_matrix: NDArray[np.float32]
    n_iterations: int


def _cost_cosine(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    na = float(np.linalg.norm(a)) + 1e-12
    nb = float(np.linalg.norm(b)) + 1e-12
    return 1.0 - float(np.dot(a, b)) / (na * nb)


def _cost_euclidean(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    return float(np.linalg.norm(a - b))


def oltw_align(
    reference: NDArray[np.floating],
    performance: NDArray[np.floating],
    search_width: int = 200,
    max_run_count: int = 3,
    metric: Metric = "euclidean",
) -> OLTWResult:
    """Align a performance against a reference using OLTW Dixon 2005.

    Args:
        reference: shape ``[M, D]``, the reference (score) features.
        performance: shape ``[N, D]``, the performance features.
        search_width: ``c`` in the paper, the windowing width of the forward
            path heuristic in frames.
        max_run_count: maximum consecutive same-direction moves before a
            forced switch. Canonical value from Dixon 2005 is ``3``.
        metric: ``"euclidean"`` (default, robust on sparse inputs like
            piano rolls) or ``"cosine"``.

    Returns:
        :class:`OLTWResult` with the per-perf-frame reference index estimate.

    Raises:
        ValueError: if reference and performance have mismatched feature
            dimension or are empty, or if ``metric`` is unknown.
    """
    if metric == "cosine":
        cost_fn = _cost_cosine
    elif metric == "euclidean":
        cost_fn = _cost_euclidean
    else:
        raise ValueError(f"unknown metric: {metric!r}")

    m_ref = reference.shape[0]
    n_perf = performance.shape[0]
    if m_ref == 0 or n_perf == 0:
        raise ValueError("reference and performance must be non-empty")
    if reference.shape[1] != performance.shape[1]:
        raise ValueError(
            f"feature dims must match, got {reference.shape[1]} vs "
            f"{performance.shape[1]}"
        )

    inf32 = np.float32(np.inf)
    d_mat = np.full((m_ref, n_perf), inf32, dtype=np.float32)
    d_mat[0, 0] = np.float32(cost_fn(reference[0], performance[0]))

    alignment = np.zeros(n_perf, dtype=np.int64)
    alignment[0] = 0

    t, j = 0, 0
    run_count = 0
    prev_dir: str | None = None
    c = search_width
    n_iter = 0

    def fill_row(t_new: int, j_lo: int, j_hi: int) -> None:
        for k in range(j_lo, j_hi + 1):
            preds = []
            if t_new > 0 and k > 0:
                preds.append(d_mat[t_new - 1, k - 1])
            if t_new > 0:
                preds.append(d_mat[t_new - 1, k])
            if k > 0:
                preds.append(d_mat[t_new, k - 1])
            local = cost_fn(reference[t_new], performance[k])
            d_mat[t_new, k] = np.float32(local) + (
                min(preds) if preds else np.float32(0.0)
            )

    def fill_col(j_new: int, t_lo: int, t_hi: int) -> None:
        for k in range(t_lo, t_hi + 1):
            preds = []
            if k > 0 and j_new > 0:
                preds.append(d_mat[k - 1, j_new - 1])
            if k > 0:
                preds.append(d_mat[k - 1, j_new])
            if j_new > 0:
                preds.append(d_mat[k, j_new - 1])
            local = cost_fn(reference[k], performance[j_new])
            d_mat[k, j_new] = np.float32(local) + (
                min(preds) if preds else np.float32(0.0)
            )

    def get_inc() -> str:
        if t < c and j < c:
            return BOTH
        if prev_dir is not None and run_count >= max_run_count:
            return COLUMN if prev_dir == ROW else ROW
        j_lo = max(0, j - c + 1)
        t_lo = max(0, t - c + 1)
        row_min = float(d_mat[t, j_lo : j + 1].min())
        col_min = float(d_mat[t_lo : t + 1, j].min())
        if row_min < col_min:
            return COLUMN
        if col_min < row_min:
            return ROW
        return BOTH

    while t < m_ref - 1 and j < n_perf - 1:
        d = get_inc()
        n_iter += 1
        new_t = t + 1 if d in (BOTH, COLUMN) else t
        new_j = j + 1 if d in (BOTH, ROW) else j

        if new_t > t:
            fill_row(new_t, max(0, j - c + 1), j)
        if new_j > j:
            t_lo = max(0, new_t - c + 1)
            fill_col(new_j, t_lo, new_t)
            window = d_mat[t_lo : new_t + 1, new_j]
            alignment[new_j] = t_lo + int(np.argmin(window))

        if d == prev_dir:
            run_count += 1
        else:
            run_count = 1
        prev_dir = d
        t, j = new_t, new_j

    while j < n_perf - 1:
        j += 1
        t_lo = max(0, t - c + 1)
        fill_col(j, t_lo, t)
        alignment[j] = t_lo + int(np.argmin(d_mat[t_lo : t + 1, j]))

    return OLTWResult(alignment=alignment, cost_matrix=d_mat, n_iterations=n_iter)
