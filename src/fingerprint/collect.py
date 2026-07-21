"""Orchestrates fingerprint probe collection, caching, and normalization."""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fingerprint.types import Fingerprint, CellData, ModelAdapter
from fingerprint.normalize import normalize_answer


def load_battery(battery_path: str | Path) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    """Loads prompt battery JSON, returns (battery_data, battery_id, battery_hash, color_lexicon)."""
    battery_path = Path(battery_path)
    if battery_path.is_dir():
        prompt_file = battery_path / "prompts.json"
        color_file = battery_path / "color-lexicon.json"
    else:
        prompt_file = battery_path
        color_file = battery_path.parent / "color-lexicon.json"

    if not prompt_file.exists():
        raise FileNotFoundError(f"Battery prompt file not found at {prompt_file}")

    with open(prompt_file, "r", encoding="utf-8") as f:
        battery_data = json.load(f)

    color_lexicon = {}
    if color_file.exists():
        with open(color_file, "r", encoding="utf-8") as f:
            color_lexicon = json.load(f)

    # Compute deterministic sha256 hash of battery prompts
    canonical_bytes = json.dumps(battery_data, sort_keys=True).encode("utf-8")
    battery_hash = hashlib.sha256(canonical_bytes).hexdigest()[:12]

    battery_id = battery_data.get("version", "v1")
    return battery_data, battery_id, battery_hash, color_lexicon


def collect(
    adapter: ModelAdapter,
    battery_path: str | Path,
    n_per_cell: int = 20,
    temperature: float = 1.0,
    claimed_model: str | None = None,
    cache_dir: str | Path | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> Fingerprint:
    """Collects empirical output distributions by probing the adapter across battery tasks."""
    battery_data, battery_id, battery_hash, color_lexicon = load_battery(battery_path)

    system_prompts = battery_data.get("system_prompts", {})
    tasks = battery_data.get("tasks", [])
    languages = battery_data.get("languages", ["en"])

    # Cell storage
    cell_records: dict[str, CellData] = {}

    total_probes = len(tasks) * len(languages) * n_per_cell
    completed_probes = 0

    cache_file: Path | None = None
    cache_data: dict[str, Any] = {}
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        adapter_name = getattr(adapter, "name", "adapter")
        cache_file = cache_dir / f"cache_{adapter_name}_{battery_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
            except Exception:
                cache_data = {}

    for task in tasks:
        task_id = task["id"]
        task_prompts = task.get("prompts", {})

        for lang in languages:
            if lang not in task_prompts:
                continue

            prompt_text = task_prompts[lang]
            sys_prompt = system_prompts.get(lang, "")
            cell_key = f"{lang}/{task_id}"

            if sys_prompt:
                full_prompt = f"{sys_prompt}\n\n{prompt_text}"
            else:
                full_prompt = prompt_text

            cell_data = cell_records.setdefault(cell_key, CellData())

            for rep in range(n_per_cell):
                cache_key = f"{cell_key}_rep{rep}_temp{temperature}"

                if cache_key in cache_data:
                    raw_text = cache_data[cache_key]["raw_text"]
                else:
                    adapter_res = adapter.complete(full_prompt, temperature=temperature)
                    raw_text = adapter_res.get("raw_text", "")
                    if cache_file:
                        cache_data[cache_key] = {
                            "raw_text": raw_text,
                            "valid": adapter_res.get("valid", False),
                            "latency_ms": adapter_res.get("latency_ms", 0.0),
                        }

                # Normalize answer
                norm_res = normalize_answer(
                    raw_text,
                    lang=lang,
                    task=task,
                    color_lexicon=color_lexicon,
                )

                if norm_res.answer_class == "valid" and norm_res.normalized:
                    token = norm_res.normalized
                    cell_data.counts[token] = cell_data.counts.get(token, 0) + 1
                    cell_data.n_valid += 1
                elif norm_res.answer_class == "refusal":
                    cell_data.n_refusal += 1
                elif norm_res.answer_class == "empty":
                    cell_data.n_empty += 1
                else:
                    cell_data.n_invalid += 1

                completed_probes += 1
                if progress_callback:
                    progress_callback(cell_key, completed_probes, total_probes)

    if cache_file and cache_data:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)

    now_iso = datetime.now(timezone.utc).isoformat()
    adapter_name = getattr(adapter, "name", "unknown")
    environment = getattr(adapter, "environment", "unknown")

    return Fingerprint(
        schema_version=1,
        battery_id=battery_id,
        battery_hash=battery_hash,
        environment=environment,
        adapter=adapter_name,
        claimed_model=claimed_model,
        temperature=temperature,
        created_at=now_iso,
        cells=cell_records,
        meta={
            "n_per_cell": n_per_cell,
            "total_probes": total_probes,
            "completed_probes": completed_probes,
        },
    )
