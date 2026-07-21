"""Orchestrates fingerprint probe collection, caching, and normalization."""

import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    max_workers: int = 8,
) -> Fingerprint:
    """Collects empirical output distributions by probing the adapter across battery tasks."""
    battery_data, battery_id, battery_hash, color_lexicon = load_battery(battery_path)

    system_prompts = battery_data.get("system_prompts", {})
    tasks = battery_data.get("tasks", [])
    languages = battery_data.get("languages", ["en"])

    # Cell storage
    cell_records: dict[str, CellData] = {}

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

    # Gather work items
    work_items: list[tuple[str, int, dict[str, Any], str, str, str]] = []
    for task in tasks:
        task_id = task["id"]
        task_prompts = task.get("prompts", {})

        for lang in languages:
            if lang not in task_prompts:
                continue

            prompt_text = task_prompts[lang]
            sys_prompt = system_prompts.get(lang, "")
            cell_key = f"{lang}/{task_id}"
            full_prompt = f"{sys_prompt}\n\n{prompt_text}" if sys_prompt else prompt_text

            cell_records.setdefault(cell_key, CellData())

            for rep in range(n_per_cell):
                cache_key = f"{cell_key}_rep{rep}_temp{temperature}"
                work_items.append((cell_key, rep, task, lang, full_prompt, cache_key))

    total_probes = len(work_items)
    completed_probes = 0

    def _process_item(item: tuple[str, int, dict[str, Any], str, str, str]) -> tuple[str, str, dict[str, Any], str, str]:
        cell_key, rep, task, lang, full_prompt, cache_key = item
        if cache_key in cache_data:
            raw_text = cache_data[cache_key]["raw_text"]
        else:
            adapter_res = adapter.complete(full_prompt, temperature=temperature)
            raw_text = getattr(adapter_res, "raw_text", "") if hasattr(adapter_res, "raw_text") else adapter_res.get("raw_text", "")
            if cache_file:
                cache_data[cache_key] = {
                    "raw_text": raw_text,
                    "valid": getattr(adapter_res, "valid", False) if hasattr(adapter_res, "valid") else adapter_res.get("valid", False),
                    "latency_ms": getattr(adapter_res, "latency_ms", 0.0) if hasattr(adapter_res, "latency_ms") else adapter_res.get("latency_ms", 0.0),
                }
        return cell_key, raw_text, task, lang, cache_key

    # Execute probes using ThreadPoolExecutor for concurrent completion fetching
    workers = min(max_workers, max(1, total_probes))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_item, item) for item in work_items]
        for future in as_completed(futures):
            cell_key, raw_text, task, lang, _ = future.result()
            cell_data = cell_records[cell_key]

            # Normalize answer
            norm_res = normalize_answer(
                raw_text,
                lang=lang,
                task=task,
                color_lexicon=color_lexicon,
            )

            if norm_res.answer_class == "valid" and norm_res.normalized:
                token = norm_res.normalized
                if task.get("category") == "color" and norm_res.color_canon:
                    token = norm_res.color_canon
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
