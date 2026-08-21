"""Caveat dataclass for structured fallback/default/degradation capture.

A Caveat records a single compromise made during export or simulation.
Collect them in a list and attach to simulation_result.json["warnings"].
Empty list means: no fallbacks, no defaults applied, no assumptions made.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

CaveatKind = Literal["fallback", "default", "degraded", "assumption"]


@dataclass(frozen=True)
class Caveat:
    """Structured record of one compromise (fallback/default/degradation/assumption).

    Attributes:
        kind:    Category — fallback (missing asset), default (substituted value),
                 degraded (simpler model used), assumption (unspecified param).
        source:  Module path or "agent" for agent-generated entries.
        message: Human-readable description of what was substituted or assumed.
    """
    kind: CaveatKind
    source: str
    message: str

    def to_dict(self) -> dict:
        """Return as a plain dict suitable for JSON serialization."""
        return asdict(self)
