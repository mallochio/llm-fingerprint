"""End-to-end collection, verification, identification, and mixture audit tests."""

from pathlib import Path
from fingerprint.adapters import MockAdapter
from fingerprint.collect import collect
from fingerprint.verify import verify, identify, mixture_report
from fingerprint.store import save_fingerprint, load_fingerprint


def test_e2e_collect_verify_identify(tmp_path: Path):
    battery_path = Path("batteries/v1")

    # 1. Collect mock fingerprints
    ad_gpt4o = MockAdapter(target_profile="gpt-4o", seed=42)
    fp_gpt4o = collect(ad_gpt4o, battery_path=battery_path, n_per_cell=10, claimed_model="gpt-4o")

    ad_claude = MockAdapter(target_profile="claude-3-5-sonnet", seed=123)
    fp_claude = collect(ad_claude, battery_path=battery_path, n_per_cell=10, claimed_model="claude-3-5-sonnet")

    # 2. Save & Load
    save_path_gpt4o = tmp_path / "gpt4o.json"
    save_fingerprint(fp_gpt4o, save_path_gpt4o)

    loaded_gpt4o = load_fingerprint(save_path_gpt4o)
    assert loaded_gpt4o.battery_id == "1.0.0"

    # 3. Verification
    v_res_same = verify(fp_gpt4o, loaded_gpt4o, tau=0.05, min_n=5)
    assert v_res_same.verified is True
    assert v_res_same.distance < 0.05

    v_res_diff = verify(fp_gpt4o, fp_claude, tau=0.05, min_n=5)
    assert v_res_diff.verified is False

    # 4. Identification
    library = [loaded_gpt4o, fp_claude]
    neighbors = identify(fp_gpt4o, library, k=2, min_n=5)
    assert len(neighbors) == 2
    assert neighbors[0].model_id == "gpt-4o"

    # 5. Mixture audit
    sessions = [fp_gpt4o, fp_claude]
    mix_res = mixture_report(sessions, library, min_n=5)
    assert mix_res.num_sessions == 2
    assert "gpt-4o" in mix_res.estimated_mixture
    assert "claude-3-5-sonnet" in mix_res.estimated_mixture
