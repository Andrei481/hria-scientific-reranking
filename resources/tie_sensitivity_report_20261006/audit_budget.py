"""Read-only audit of the archived 1,024-token tie-sensitivity experiment.

Usage:
    python audit_budget.py PATH_TO_BACKUP_DIRECTORY

Verifies both backup archives and each full evaluation's checksums. Reports
whether appending to a truncated response could affect the measured top-k
scores under the saved first-occurrence parser. Does not run model inference.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile


SEEDS = (42, 2026, 3407)
POLICIES = ("deterministic", "epoch_random")
RETRIEVERS = ("bm25", "grit", "qwen3")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_response(row):
    candidates = row["pre_reranked"]
    require(len(candidates) == 100, "Unexpected candidate count")
    require(len(set(candidates)) == len(candidates), "Duplicate candidate documents")
    mentions = [int(value) for value in re.findall(r"\[(\d+)\]", row["raw_response"])]
    valid = list(dict.fromkeys(value for value in mentions if 1 <= value <= len(candidates)))
    missing = [value for value in range(1, len(candidates) + 1) if value not in valid]
    repaired = [candidates[value - 1] for value in valid + missing]
    require(repaired == row["retrieved"], "Raw output and saved repaired ranking disagree")
    require(len(missing) == row["parser"]["missing_ids"], "Missing-ID audit disagrees")
    relevant = set(row["corpusids"])
    # A nonempty relevance set appearing exactly in the first |relevant|
    # emitted positions has perfect recall/nDCG, protected against extensions.
    perfect_prefix = (
        bool(relevant)
        and len(valid) >= len(relevant)
        and set(repaired[:len(relevant)]) == relevant
    )
    return valid, perfect_prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup", type=Path)
    args = parser.parse_args()
    expected = {}
    for line in (args.backup / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            expected[name.lstrip("*")] = digest
    names = {"report_and_resources.tar.gz", "final_adapter.tar.gz"}
    require(set(expected) == names, "Unexpected archive names in checksum manifest")
    for name in sorted(names):
        require(digest_file(args.backup / name) == expected[name], f"Archive hash mismatch: {name}")

    prefixes = {
        (policy, seed): f"tie_sensitivity_artifacts/runs_disk_safe/{policy}_seed{seed}/"
        for policy in POLICIES for seed in SEEDS
    }
    wanted = set()
    for prefix in prefixes.values():
        wanted.add(prefix + "training_manifest.json")
        wanted.update(prefix + "evaluation/" + name for name in (
            "protocol.json", "complete.json", "bm25.jsonl", "grit.jsonl", "qwen3.jsonl"
        ))
    payloads = {}
    with tarfile.open(args.backup / "report_and_resources.tar.gz", "r:gz") as archive:
        for member in archive:
            if member.name in wanted:
                require(member.isfile(), f"Expected regular archived file: {member.name}")
                require(member.name not in payloads, f"Duplicate archive member: {member.name}")
                payloads[member.name] = archive.extractfile(member).read()
    require(set(payloads) == wanted, "Archive is missing required evaluation records")

    groups, exceptions = [], []
    total = capped_total = capped_under20 = potentially_affected = 0
    protected_short = 0
    for (policy, seed), prefix in prefixes.items():
        training_bytes = payloads[prefix + "training_manifest.json"]
        training = json.loads(training_bytes)
        require(training["status"] == "complete", "Training is incomplete")
        require(training["config"]["policy"] == policy and training["config"]["seed"] == seed,
                "Training identity disagrees with directory")
        protocol_bytes = payloads[prefix + "evaluation/protocol.json"]
        protocol = json.loads(protocol_bytes)
        complete = json.loads(payloads[prefix + "evaluation/complete.json"])
        require(hashlib.sha256(protocol_bytes).hexdigest() == complete["protocol_sha256"],
                "Evaluation protocol hash mismatch")
        require(hashlib.sha256(training_bytes).hexdigest() == protocol["training_manifest_sha256"],
                "Training manifest hash mismatch")
        require(protocol["max_tokens"] == 1024 and protocol["limit"] == 0,
                "Unexpected evaluation budget or smoke protocol")
        require(protocol["temperature"] == 0 and protocol["top_p"] == 1
                and protocol["generation_seed"] == 0, "Unexpected decoding settings")
        for retriever in RETRIEVERS:
            content = payloads[prefix + f"evaluation/{retriever}.jsonl"]
            require(hashlib.sha256(content).hexdigest() == complete["results"][retriever],
                    "Evaluation result hash mismatch")
            rows = [json.loads(line) for line in content.splitlines() if line.strip()]
            require(len(rows) == 597 and len({row["qid"] for row in rows}) == 597,
                    "Unexpected query count or duplicate query IDs")
            cap_counts, all_counts = [], []
            stopped_under20 = 0
            for row in rows:
                valid, perfect_prefix = inspect_response(row)
                count = len(valid)
                all_counts.append(count)
                total += 1
                require(row["finish_reason"] in ("stop", "length"), "Unexpected finish reason")
                if row["finish_reason"] == "length":
                    capped_total += 1
                    cap_counts.append(count)
                    if count < 20:
                        capped_under20 += 1
                        protected_short += int(perfect_prefix)
                        potentially_affected += int(not perfect_prefix)
                        exceptions.append({
                            "policy": policy, "seed": seed, "retriever": retriever,
                            "qid": row["qid"], "valid_ids": valid,
                            "relevant_documents": len(set(row["corpusids"])),
                            "perfect_relevance_prefix": perfect_prefix,
                        })
                elif count < 20:
                    stopped_under20 += 1
            groups.append({
                "policy": policy, "seed": seed, "retriever": retriever,
                "responses": len(rows), "length_finished": len(cap_counts),
                "min_valid_ids_if_length_finished": min(cap_counts, default=None),
                "min_valid_ids_all": min(all_counts), "stopped_under20": stopped_under20,
            })
    print(json.dumps({
        "verified_archive_sha256": expected,
        "responses": total, "length_finished": capped_total,
        "length_finished_with_at_least20_valid_ids": capped_total - capped_under20,
        "length_finished_under20": capped_under20,
        "short_truncated_responses_protected_by_perfect_prefix": protected_short,
        "truncated_responses_with_potentially_changed_top20_metrics": potentially_affected,
        "condition": "Appending to the same response; preserves emitted prefix and first-occurrence parser. This is not a 2048-token inference experiment.",
        "exceptions": exceptions, "groups": groups,
    }, indent=2))


if __name__ == "__main__":
    main()
