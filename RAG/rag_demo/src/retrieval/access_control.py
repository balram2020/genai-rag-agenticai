from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ..models import AccessLevel, ChunkRecord, ROLE_ALLOWED_LEVELS, UserRole


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    reason: str


def allowed_levels_for_role(role: UserRole) -> Tuple[AccessLevel, ...]:
    return ROLE_ALLOWED_LEVELS[role]


def can_access(role: UserRole, chunk: ChunkRecord) -> AccessDecision:
    allowed = chunk.access_level in ROLE_ALLOWED_LEVELS[role]
    if allowed:
        return AccessDecision(True, f"role={role.value} allows {chunk.access_level.value}")
    return AccessDecision(False, f"role={role.value} does NOT allow {chunk.access_level.value}")


def filter_chunks_by_role(
    role: UserRole,
    chunks: Sequence[ChunkRecord],
    max_debug_examples: int = 5,
) -> Tuple[List[ChunkRecord], int, List[Dict]]:
    allowed: List[ChunkRecord] = []
    filtered_out = 0
    examples: List[Dict] = []
    for c in chunks:
        decision = can_access(role, c)
        if decision.allowed:
            allowed.append(c)
        else:
            filtered_out += 1
            if len(examples) < max_debug_examples:
                examples.append(
                    {
                        "chunk_uid": c.chunk_uid,
                        "document_name": c.document_name,
                        "access_level": c.access_level.value,
                        "reason": decision.reason,
                    }
                )
    return allowed, filtered_out, examples

