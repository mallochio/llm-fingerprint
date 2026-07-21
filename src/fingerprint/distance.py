"""Jensen-Shannon Divergence and fingerprint distance calculations."""

import math
from fingerprint.types import Fingerprint, DistanceResult, CellData


def jsd_base2(p: dict[str, float], q: dict[str, float]) -> float:
    """Calculates base-2 Jensen-Shannon Divergence between discrete distributions p and q.
    
    Result is bounded in [0, 1].
    """
    support = set(p.keys()).union(set(q.keys()))
    if not support:
        return 0.0

    d = 0.0
    for x in support:
        px = p.get(x, 0.0)
        qx = q.get(x, 0.0)
        mx = 0.5 * (px + qx)
        if mx > 0:
            if px > 0:
                d += 0.5 * px * math.log2(px / mx)
            if qx > 0:
                d += 0.5 * qx * math.log2(qx / mx)

    return max(0.0, min(1.0, d))


def cell_to_prob(cell: CellData) -> dict[str, float] | None:
    """Converts raw counts in a cell to a probability distribution if valid count > 0."""
    if cell.n_valid <= 0 or not cell.counts:
        return None
    total = float(cell.n_valid)
    return {k: v / total for k, v in cell.counts.items()}


def distance(
    fa: Fingerprint,
    fb: Fingerprint,
    min_n: int = 10,
    warn_cross_environment: bool = True,
) -> DistanceResult:
    """Computes mean JSD between two fingerprints over cells with n_valid >= min_n."""
    all_cell_keys = set(fa.cells.keys()).union(set(fb.cells.keys()))

    cell_distances: dict[str, float] = {}
    eligible_count = 0

    for cell_key in sorted(all_cell_keys):
        cell_a = fa.cells.get(cell_key)
        cell_b = fb.cells.get(cell_key)

        if not cell_a or not cell_b:
            continue
        if cell_a.n_valid < min_n or cell_b.n_valid < min_n:
            continue

        prob_a = cell_to_prob(cell_a)
        prob_b = cell_to_prob(cell_b)

        if prob_a is None or prob_b is None:
            continue

        cell_jsd = jsd_base2(prob_a, prob_b)
        cell_distances[cell_key] = cell_jsd
        eligible_count += 1

    if eligible_count == 0:
        mean_d = 1.0  # Default maximum distance when no overlapping cells
    else:
        mean_d = sum(cell_distances.values()) / eligible_count

    return DistanceResult(
        distance=round(mean_d, 6),
        eligible_cells=eligible_count,
        total_cells=len(all_cell_keys),
        cell_distances={k: round(v, 6) for k, v in cell_distances.items()},
    )
