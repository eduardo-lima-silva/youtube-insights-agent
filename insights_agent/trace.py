"""Tipos do rastro de execução do agente (serializáveis em JSON)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TraceStep:
    kind: str                 # "llm" | "tool" | "guard" | "error"
    name: str                 # ex.: "run_sql", "score_titles", "claude-opus-5-5"
    input: dict[str, Any] = field(default_factory=dict)
    output_preview: str = ""  # texto curto exibido no painel de rastro
    ok: bool = True
    elapsed_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)   # ex.: {"sql_validated": ..., "row_count": ...}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    llm_calls: int = 0

    def add(self, u: Any) -> None:
        """Soma um objeto `usage` da API (campos ausentes contam como 0)."""
        self.input_tokens += getattr(u, "input_tokens", 0) or 0
        self.output_tokens += getattr(u, "output_tokens", 0) or 0
        self.cache_read_input_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
        self.cache_creation_input_tokens += getattr(u, "cache_creation_input_tokens", 0) or 0
        self.llm_calls += 1


@dataclass
class AgentResult:
    question: str
    answer: str = ""
    status: str = "ok"        # ok | refused_by_model | max_tool_calls | error
    steps: list[TraceStep] = field(default_factory=list)
    charts: list[dict[str, Any]] = field(default_factory=list)   # specs Altair (chart.to_dict())
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    latency_s: float = 0.0
    model: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentResult":
        d = dict(d)
        d["steps"] = [TraceStep(**s) for s in d.get("steps", [])]
        d["usage"] = Usage(**d.get("usage", {}))
        return cls(**d)
