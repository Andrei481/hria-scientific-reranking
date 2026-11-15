"""Shared controls for the tie-sensitivity experiment (standard library only)."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re

POLICIES = ("deterministic", "fixed_random", "epoch_random")
FROZEN_SHA256 = "7c649c816c78778777afcddc011dafd345a6847d2fd6d4e082b58bad75e5bc4f"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts):
    material = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def example_id(row):
    return f"{row['source_row_id']}:{row['variant_id']}"


def audit_rows(rows):
    if not rows:
        raise ValueError("Empty supervision")
    keys = set()
    lengths, positives, grades = Counter(), [], Counter()
    for row in rows:
        key = example_id(row)
        if key in keys:
            raise ValueError(f"Duplicate example ID: {key}")
        keys.add(key)
        docs, labels, ids = row["docs"], row["labels"], row["candidate_ids"]
        if not isinstance(row["query"], str) or not row["query"].strip():
            raise ValueError(f"Empty query: {key}")
        if not docs or not len(docs) == len(labels) == len(ids):
            raise ValueError(f"Unaligned or empty list: {key}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate candidate ID: {key}")
        if any(type(label) is not int or label not in (0, 1, 2) for label in labels):
            raise ValueError(f"Invalid relevance grade: {key}")
        if not any(labels) or any(not isinstance(d, str) or not d.strip() for d in docs):
            raise ValueError(f"Missing text or no positives: {key}")
        lengths[str(len(docs))] += 1
        grades.update(labels)
        positives.append(sum(label > 0 for label in labels))
    return {
        "examples": len(rows), "list_size_histogram": dict(lengths),
        "grade_counts": dict(grades), "max_positives_observed": max(positives),
        "lists_above_five_positives": sum(n > 5 for n in positives),
    }


def target_indices(labels, policy, seed, key, epoch):
    if policy not in POLICIES or not labels or any(type(y) is not int or y not in (0, 1, 2) for y in labels):
        raise ValueError("Unknown tie policy or invalid relevance grades")
    rng = random.Random(stable_seed("ties", seed, key, epoch if policy == "epoch_random" else 0))
    order = []
    for grade in (2, 1, 0):
        group = [i + 1 for i, value in enumerate(labels) if value == grade]
        if policy != "deterministic":
            rng.shuffle(group)
        order.extend(group)
    return order


def epoch_schedule(count, epochs, seed):
    """Explicit immutable epoch IDs avoid worker/callback prefetch ambiguity."""
    if count <= 0 or epochs <= 0:
        raise ValueError("Positive count and epoch count required")
    schedule = []
    for epoch in range(epochs):
        order = list(range(count))
        random.Random(stable_seed("data_order", seed, epoch)).shuffle(order)
        schedule.extend((epoch, index) for index in order)
    return schedule


def prompt_text(query, documents):
    passages = "\n\n---\n\n".join(
        f"[{i}] {' '.join(document.split()[:300])}"
        for i, document in enumerate(documents, 1)
    )
    return (
        "You are an expert academic paper reranker. "
        f"Your task is to re-order the given list of passages (from [1] to [{len(documents)}]) "
        "based on their relevance to the query. Respond with only the ranking and nothing else.\n\n"
        "Example output for 8 passages:\n"
        "[2] > [5] > [4] > [8] > [6] > [1] > [3] > [7]\n\n"
        f"Query: {query}\n\nPassages:\n{passages}\n\nYour ranking (most to least relevant):"
    )


def encode_example(row, tokenizer, policy, seed, epoch, max_length):
    order = target_indices(row["labels"], policy, seed, example_id(row), epoch)
    response = " > ".join(f"[{i}]" for i in order)
    messages = [{"role": "user", "content": prompt_text(row["query"], row["docs"])}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": response}],
        tokenize=False, add_generation_prompt=False,
    )
    prefix_ids = tokenizer(prefix, add_special_tokens=False, truncation=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False, truncation=False)["input_ids"]
    if ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("Chat tokenizer prefix mismatch; response mask would be unsafe")
    if len(ids) > max_length:
        raise ValueError(f"Example {example_id(row)} needs {len(ids)} tokens, exceeds {max_length}; no silent truncation")
    if len(ids) <= len(prefix_ids):
        raise ValueError("No supervised response tokens")
    return {
        "input_ids": ids, "attention_mask": [1] * len(ids),
        "labels": [-100] * len(prefix_ids) + ids[len(prefix_ids):],
    }


def check_data_hash(path, manifest):
    digest = sha256_file(path)
    if digest != manifest["data_sha256"]:
        raise ValueError("Frozen training file does not match its manifest")
    return digest


def parse_ranking(text, count):
    if count < 1:
        raise ValueError("Candidate list is empty")
    raw = [int(s) for s in re.findall(r"\[(\d+)\]", text)]
    valid, seen = [], set()
    duplicate = invalid = 0
    for value in raw:
        if not 1 <= value <= count:
            invalid += 1
        elif value in seen:
            duplicate += 1
        else:
            seen.add(value)
            valid.append(value - 1)
    missing = count - len(valid)
    audit = {
        "exact_permutation": not missing and not duplicate and not invalid,
        "format_only": bool(re.fullmatch(r"\s*\[\d+\](?:\s*>\s*\[\d+\])*\s*", text)),
        "duplicate_ids": duplicate, "invalid_ids": invalid, "missing_ids": missing,
        "fallback": not valid,
    }
    return valid + [i for i in range(count) if i + 1 not in seen], audit


def litsearch_metrics(retrieved, relevant):
    relevant = set(relevant)
    result = {}
    for k in (5, 20):
        result[f"R@{k}"] = len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0
        dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(retrieved[:k]) if d in relevant)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
        result[f"nDCG@{k}"] = dcg / ideal if ideal else 0.0
    return result


def source_hashes():
    return {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py")) if not p.name.startswith("test_")}
