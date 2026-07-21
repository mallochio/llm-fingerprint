"""Verification, identification, and router mixture auditing."""

from typing import Sequence
from collections import Counter

from fingerprint.types import (
    Fingerprint,
    VerifyResult,
    Neighbor,
    MixtureResult,
    SessionReport,
)
from fingerprint.distance import distance


def verify(
    f_unknown: Fingerprint,
    f_claim: Fingerprint,
    tau: float = 0.05,
    min_n: int = 10,
) -> VerifyResult:
    """Verifies whether f_unknown matches f_claim within threshold tau."""
    env_match = (
        f_unknown.environment == f_claim.environment
        and f_unknown.environment != "unknown"
    )
    dist_res = distance(f_unknown, f_claim, min_n=min_n)

    claimed_name = f_claim.claimed_model or f_claim.adapter or "claimed_model"
    is_verified = dist_res.distance < tau

    return VerifyResult(
        claimed_model=claimed_name,
        distance=dist_res.distance,
        threshold=tau,
        verified=is_verified,
        environment_match=env_match,
        eligible_cells=dist_res.eligible_cells,
        environment_unknown=f_unknown.environment,
        environment_claim=f_claim.environment,
    )


def identify(
    f_unknown: Fingerprint,
    library: Sequence[Fingerprint] | dict[str, Fingerprint],
    k: int = 5,
    min_n: int = 10,
    filter_environment: bool = False,
) -> list[Neighbor]:
    """Identifies top-k nearest reference models for f_unknown."""
    if isinstance(library, dict):
        fps = list(library.values())
    else:
        fps = list(library)

    neighbors: list[Neighbor] = []
    for ref_fp in fps:
        if filter_environment and ref_fp.environment != f_unknown.environment:
            continue

        model_name = ref_fp.claimed_model or ref_fp.adapter or "unknown_ref"
        dist_res = distance(f_unknown, ref_fp, min_n=min_n)
        neighbors.append(
            Neighbor(
                model_id=model_name,
                distance=dist_res.distance,
                environment=ref_fp.environment,
                eligible_cells=dist_res.eligible_cells,
            )
        )

    neighbors.sort(key=lambda n: n.distance)
    return neighbors[:k]


def mixture_report(
    sessions: Sequence[Fingerprint],
    library: Sequence[Fingerprint] | dict[str, Fingerprint],
    k_neighbors: int = 3,
    min_n: int = 1,
) -> MixtureResult:
    """Audits router / Auto modes as mixtures over multiple independent sessions."""
    if not sessions:
        return MixtureResult(
            num_sessions=0,
            session_pairwise_mean_jsd=0.0,
            estimated_mixture={},
            sessions=[],
        )

    # 1. Compute session pairwise JSD
    pairwise_distances = []
    for i in range(len(sessions)):
        for j in range(i + 1, len(sessions)):
            d = distance(sessions[i], sessions[j], min_n=min_n)
            pairwise_distances.append(d.distance)

    mean_pairwise = (
        sum(pairwise_distances) / len(pairwise_distances)
        if pairwise_distances
        else 0.0
    )

    # 2. Identify top neighbor for each session
    session_reports: list[SessionReport] = []
    top_matches = []

    for idx, s in enumerate(sessions):
        nbrs = identify(s, library, k=k_neighbors, min_n=min_n)
        top_match = nbrs[0].model_id if nbrs else "unknown"
        top_dist = nbrs[0].distance if nbrs else 1.0

        top_matches.append(top_match)
        session_reports.append(
            SessionReport(
                session_id=idx + 1,
                top_match=top_match,
                top_distance=top_dist,
                neighbors=nbrs,
            )
        )

    # 3. Estimate mixture weights
    match_counts = Counter(top_matches)
    total_sess = len(sessions)
    mixture = {
        model: round(count / total_sess, 4)
        for model, count in match_counts.most_common()
    }

    return MixtureResult(
        num_sessions=total_sess,
        session_pairwise_mean_jsd=round(mean_pairwise, 6),
        estimated_mixture=mixture,
        sessions=session_reports,
    )
