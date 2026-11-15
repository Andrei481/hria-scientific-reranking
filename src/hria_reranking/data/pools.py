"""Deterministic pool-level augmentation extracted from the notebooks."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PoolConfig:
    """Candidate-pool sampling parameters used by the training recipe."""

    variants: int = 3
    target_size: int = 100
    max_positives: int = 5
    positive_mode: str = "cycle"
    hard_negative_top: int | None = 100
    max_retries: int = 3000

    def __post_init__(self) -> None:
        if self.variants < 0:
            raise ValueError("variants must be non-negative")
        if self.target_size <= 0:
            raise ValueError("target_size must be positive")
        if self.max_positives < 0:
            raise ValueError("max_positives must be non-negative")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.hard_negative_top is not None and self.hard_negative_top < 0:
            raise ValueError("hard_negative_top must be non-negative or None")
        if self.positive_mode not in {"cycle", "only_2", "only_1", "mixed", "any"}:
            raise ValueError(
                "positive_mode must be cycle, only_2, only_1, mixed, or any"
            )


def stable_seed(row_id: Any, query_text: str) -> int:
    """Derive the notebook's stable 32-bit seed from row identity and query."""
    digest = hashlib.md5(f"{row_id}||{query_text}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _config_value(
    config: Any, name: str, aliases: tuple[str, ...], default: Any
) -> Any:
    if isinstance(config, Mapping):
        for key in (name, *aliases):
            if key in config:
                return config[key]
        return default
    for key in (name, *aliases):
        if hasattr(config, key):
            return getattr(config, key)
    return default


def _coerce_config(config: PoolConfig | Mapping[str, Any] | Any) -> PoolConfig:
    if isinstance(config, PoolConfig):
        return config
    hard_negative_top = _config_value(
        config, "hard_negative_top", ("hard_neg_top",), 100
    )
    return PoolConfig(
        variants=int(_config_value(config, "variants", ("n_variants",), 3)),
        target_size=int(_config_value(config, "target_size", (), 100)),
        max_positives=int(_config_value(config, "max_positives", (), 5)),
        positive_mode=str(_config_value(config, "positive_mode", (), "cycle")),
        hard_negative_top=(
            None if hard_negative_top is None else int(hard_negative_top)
        ),
        max_retries=int(
            _config_value(config, "max_retries", ("max_total_tries",), 3000)
        ),
    )


def _signature(selected: list[tuple[Any, int]]) -> tuple[int, ...]:
    return tuple(sorted(int(document_id) for document_id, _ in selected))


def _sample_positives(
    rng: random.Random,
    mode: str,
    score_one: list[tuple[Any, int]],
    score_two: list[tuple[Any, int]],
    all_positives: list[tuple[Any, int]],
    maximum: int,
) -> list[tuple[Any, int]]:
    if mode == "only_2":
        count = min(len(score_two), maximum)
        return rng.sample(score_two, count) if count else []
    if mode == "only_1":
        count = min(len(score_one), maximum)
        return rng.sample(score_one, count) if count else []
    if mode == "mixed":
        score_two_count = min(len(score_two), maximum // 2)
        score_one_count = min(len(score_one), maximum - score_two_count)
        return (
            (rng.sample(score_two, score_two_count) if score_two_count else [])
            + (rng.sample(score_one, score_one_count) if score_one_count else [])
        )
    count = min(len(all_positives), maximum)
    return rng.sample(all_positives, count) if count else []


def sample_unique_pools(
    instance: Mapping[str, Any], config: PoolConfig | Mapping[str, Any] | Any
) -> list[list[tuple[Any, int]]]:
    """Sample deterministic, fixed-length, set-unique document pools.

    Document order is shuffled only after set uniqueness has been established.
    Positive selection cycles through ``only_2``, ``only_1``, ``mixed``, and
    three ``any`` attempts, matching the canonical notebook.
    """
    resolved = _coerce_config(config)
    candidate_pool = list(instance["candidate_pool"])
    candidate_scores = list(instance["candidate_scores"])
    if len(candidate_pool) != len(candidate_scores):
        raise ValueError("candidate_pool and candidate_scores must have equal length")
    unique_document_ids = {int(document_id) for document_id in candidate_pool}
    if len(unique_document_ids) != len(candidate_pool):
        raise ValueError("candidate_pool document IDs must be unique")
    if not candidate_pool or resolved.variants == 0:
        return []

    scored = [
        (document_id, int(score))
        for document_id, score in zip(candidate_pool, candidate_scores)
    ]
    negatives = [item for item in scored if item[1] == 0]
    score_one = [item for item in scored if item[1] == 1]
    score_two = [item for item in scored if item[1] >= 2]
    all_positives = score_two + score_one
    if resolved.hard_negative_top is not None:
        negatives = negatives[: int(resolved.hard_negative_top)]

    if resolved.target_size > len(negatives) + min(
        len(all_positives), resolved.max_positives
    ):
        raise ValueError(
            "target_size cannot be met without exceeding the positive cap and "
            "hard-negative limit"
        )

    modes = ["only_2", "only_1", "mixed", "any", "any", "any"]
    if resolved.positive_mode != "cycle":
        modes = [resolved.positive_mode]

    row_id = instance.get("row_id", instance.get("index", 0))
    query_text = str(instance["query_text"])
    rng = random.Random(stable_seed(row_id, query_text))
    seen: set[tuple[int, ...]] = set()
    pools: list[list[tuple[Any, int]]] = []

    tries = 0
    while len(pools) < resolved.variants and tries < resolved.max_retries:
        mode = modes[tries % len(modes)]
        tries += 1
        positives = _sample_positives(
            rng,
            mode,
            score_one,
            score_two,
            all_positives,
            min(resolved.max_positives, resolved.target_size),
        )
        if not positives and all_positives and resolved.max_positives > 0:
            positives = [rng.choice(all_positives)]

        needed = resolved.target_size - len(positives)
        negative_count = min(len(negatives), needed)
        selected_negatives = (
            rng.sample(negatives, negative_count) if negative_count else []
        )
        selected = positives + selected_negatives

        if len(selected) < resolved.target_size:
            selected_ids = {document_id for document_id, _ in selected}
            positive_capacity = resolved.max_positives - len(positives)
            remaining_positives = [
                item for item in all_positives if item[0] not in selected_ids
            ][:positive_capacity]
            selected.extend(
                remaining_positives[: resolved.target_size - len(selected)]
            )

        if len(selected) != resolved.target_size:
            continue
        signature = _signature(selected)
        if signature in seen:
            continue

        seen.add(signature)
        rng.shuffle(selected)
        pools.append(selected)

    return pools
