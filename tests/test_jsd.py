"""Unit tests for Jensen-Shannon Divergence and fingerprint distance math."""

from fingerprint.types import Fingerprint, CellData
from fingerprint.distance import jsd_base2, distance


def test_jsd_identical():
    p = {"blue": 0.6, "red": 0.4}
    q = {"blue": 0.6, "red": 0.4}
    assert abs(jsd_base2(p, q)) < 1e-6


def test_jsd_disjoint():
    p = {"blue": 1.0}
    q = {"red": 1.0}
    assert abs(jsd_base2(p, q) - 1.0) < 1e-6


def test_fingerprint_distance_matching():
    fp_a = Fingerprint(
        environment="mock-cli",
        adapter="mock_a",
        cells={
            "en/color-random": CellData(counts={"blue": 10, "red": 10}, n_valid=20),
            "en/num10-random": CellData(counts={"7": 15, "3": 5}, n_valid=20),
        },
    )

    fp_b = Fingerprint(
        environment="mock-cli",
        adapter="mock_b",
        cells={
            "en/color-random": CellData(counts={"blue": 10, "red": 10}, n_valid=20),
            "en/num10-random": CellData(counts={"7": 15, "3": 5}, n_valid=20),
        },
    )

    res = distance(fp_a, fp_b, min_n=10)
    assert res.eligible_cells == 2
    assert res.distance < 1e-4


def test_fingerprint_distance_min_n_filtering():
    fp_a = Fingerprint(
        cells={
            "en/color-random": CellData(counts={"blue": 5}, n_valid=5),  # Below min_n=10
            "en/num10-random": CellData(counts={"7": 10}, n_valid=10),
        }
    )
    fp_b = Fingerprint(
        cells={
            "en/color-random": CellData(counts={"blue": 5}, n_valid=5),
            "en/num10-random": CellData(counts={"7": 10}, n_valid=10),
        }
    )

    res = distance(fp_a, fp_b, min_n=10)
    assert res.eligible_cells == 1
