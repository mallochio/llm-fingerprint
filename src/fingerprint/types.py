"""Core data structures and protocols for llm-fingerprint."""

from typing import TypedDict, Protocol, Any, Literal
from pydantic import BaseModel, Field


class AdapterResult(TypedDict):
    raw_text: str
    extracted_token: str | None
    valid: bool
    invalid_reason: str | None
    latency_ms: float
    exit_code: int | None
    stdout: str
    stderr: str
    meta: dict[str, Any]


class ModelAdapter(Protocol):
    name: str
    environment: str

    def complete(self, prompt: str, *, temperature: float = 1.0) -> AdapterResult:
        ...


class NormalizationResult(BaseModel):
    normalized: str | None = None
    answer_class: Literal["valid", "invalid", "refusal", "empty"] = "invalid"
    color_canon: str | None = None


class CellData(BaseModel):
    counts: dict[str, int] = Field(default_factory=dict)
    n_valid: int = 0
    n_invalid: int = 0
    n_refusal: int = 0
    n_empty: int = 0


class Fingerprint(BaseModel):
    schema_version: int = 1
    battery_id: str = "v1"
    battery_hash: str = ""
    environment: str = "unknown"
    adapter: str = "unknown"
    claimed_model: str | None = None
    temperature: float = 1.0
    created_at: str = ""
    cells: dict[str, CellData] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)


class DistanceResult(BaseModel):
    distance: float
    eligible_cells: int
    total_cells: int
    cell_distances: dict[str, float] = Field(default_factory=dict)


class VerifyResult(BaseModel):
    claimed_model: str
    distance: float
    threshold: float
    verified: bool
    environment_match: bool
    eligible_cells: int
    environment_unknown: str | None = None
    environment_claim: str | None = None


class Neighbor(BaseModel):
    model_id: str
    distance: float
    environment: str
    eligible_cells: int


class SessionReport(BaseModel):
    session_id: int
    top_match: str
    top_distance: float
    neighbors: list[Neighbor]


class MixtureResult(BaseModel):
    num_sessions: int
    session_pairwise_mean_jsd: float
    estimated_mixture: dict[str, float]
    sessions: list[SessionReport] = Field(default_factory=list)
