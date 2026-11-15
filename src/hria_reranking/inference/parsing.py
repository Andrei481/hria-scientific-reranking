"""Correction of noisy listwise rankings into complete permutations."""

from __future__ import annotations

import re
from typing import Any


_BRACKETED_IDENTIFIER = re.compile(r"\[(\d+)\]")


def parse_permutation(text: Any, candidate_count: int) -> list[int]:
    """Return a complete one-based permutation parsed from model output.

    Only bracketed identifiers are accepted.  Out-of-bounds identifiers and
    stable duplicates are discarded, then omitted identifiers are appended in
    original order.  Consequently malformed or empty output falls back to the
    original candidate order.
    """
    if candidate_count < 0:
        raise ValueError("candidate_count must be non-negative")

    seen: set[int] = set()
    parsed: list[int] = []
    if text is not None:
        for match in _BRACKETED_IDENTIFIER.findall(str(text)):
            identifier = int(match)
            if 1 <= identifier <= candidate_count and identifier not in seen:
                seen.add(identifier)
                parsed.append(identifier)

    parsed.extend(
        identifier
        for identifier in range(1, candidate_count + 1)
        if identifier not in seen
    )
    return parsed
