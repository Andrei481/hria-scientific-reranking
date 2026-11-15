"""Run from the repository root to verify recovered Gemma V2 predictions."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "CSFCube/eval_scripts"))
import ranking_eval as evaluator

RUNS = {
    "base": ("runs_gemma_3_27b_base_19102026_vllm_full_list_4096", "gemma_3_27b_base_19102026_full_list-4096"),
    "FT": ("runs_gemma_3_27b_20260218-181123_vllm_full_list_4096", "gemma_3_27b_20260218-181123_full_list-4096"),
}
COLUMNS = ["r_precision", "precision", "recall", "ndcg", "ndcg@20", "ndcg%20"]
FACETS = ["background", "method", "result"]
hashes = {}


def record(path):
    hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(path):
    return json.loads(record(path).read_text(encoding="utf-8"))


def read_csv(path):
    with record(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {r["paper_id"]: r for r in rows}
    assert len(indexed) == len(rows), f"Duplicate CSV cases: {path}"
    return indexed


def metric_lines(text):
    return [s.strip() for s in text.splitlines()
            if re.fullmatch(r"\d+\.\d+(?:, \d+\.\d+){5}", s.strip())]


report = {}
for label, (folder, experiment) in RUNS.items():
    directory = ROOT / "CSFCube/eval_scripts" / folder
    prefix = f"test-pid2pool-csfcube-{experiment}"
    saved = read_csv(directory / f"{prefix}-all-perq.csv")
    computed = {}
    total_candidates = 0
    max_error = 0.0
    aggregate_lines = []
    aggregate_values = {}
    for facet in FACETS:
        gold = read_json(ROOT / f"CSFCube/test-pid2anns-csfcube-{facet}.json")
        ranked = read_json(directory / f"{prefix}-{facet}-ranked.json")
        facet_saved = read_csv(directory / f"{prefix}-{facet}-perq.csv")
        assert set(gold) == set(ranked), f"Query coverage differs: {label}/{facet}"
        for pid, pool in gold.items():
            ids = [row[0] for row in ranked[pid]]
            assert len(ids) == len(set(ids)) == len(pool["cands"])
            assert set(ids) == set(pool["cands"]), f"Candidate mismatch: {pid}/{facet}"
            grades = dict(zip(pool["cands"], pool["relevance_adju"]))
            key = f"{pid}_{facet}"
            computed[key] = evaluator.compute_metrics([grades[p] for p in ids], 20, 2)
            total_candidates += len(ids)
            for column, value in computed[key].items():
                if column == "ndcg@20":  # Not emitted in original per-query CSVs.
                    continue
                for rows in (saved, facet_saved):
                    error = abs(value - float(rows[key][column]))
                    max_error = max(max_error, error)
                    assert math.isclose(value, float(rows[key][column]), abs_tol=1e-12, rel_tol=0), (key, column)
        assert set(facet_saved) == {f"{p}_{facet}" for p in gold}
    assert set(computed) == set(saved)
    assert len(computed) == 50
    parents = {key.rsplit("_", 1)[0] for key in computed}
    assert len(parents) == 34
    for facet in FACETS + ["all"]:
        selected = {k: v for k, v in computed.items() if facet == "all" or k.endswith("_" + facet)}
        lines = []
        aggregate_values[facet] = {}
        for split in ("dev", "test"):
            values = evaluator.aggregate_metrics_crossval(selected, split_str=split, facet_str=facet)
            aggregate_values[facet][split] = values
            lines.append(", ".join(f"{values[c]:.4f}" for c in COLUMNS))
        log = record(directory / f"metrics_{facet}.txt")
        assert lines == metric_lines(log.read_text()), f"New log mismatch: {label}/{facet}"
        aggregate_lines.extend(lines)
    original = record(ROOT / f"reproducibility_package/results/csfcube/gemma3_27b/{label}/results.txt")
    assert aggregate_lines == metric_lines(original.read_text()), f"Original V2 log mismatch: {label}"
    report[label] = {
        "source_csv": (directory / f"{prefix}-all-perq.csv").relative_to(ROOT).as_posix(),
        "cases": len(computed), "parent_papers": len(parents),
        "total_candidates": total_candidates, "max_per_case_csv_error": max_error,
        "all_rankings_complete_unique_permutations": True,
        "all_original_V2_log_aggregates_match_at_four_decimals": True,
        "aggregates": aggregate_values,
        "pooled": {c: statistics.mean(v[c] for v in computed.values()) for c in COLUMNS},
    }
for name in ("ranking_eval.py", "rank_metrics.py"):
    record(ROOT / "CSFCube/eval_scripts" / name)
analysis_config = read_json(ROOT / "configs/paired_statistical_uncertainty_csfcube_v2_20261020.json")
for contrast in analysis_config["contrasts"]:
    for role in ("reference", "candidate"):
        record(ROOT / contrast[role])
record(ROOT / "tools/paired_statistical_uncertainty.py")
record(ROOT / "tools/litsearch_document_disjoint.py")
record(Path(__file__).resolve())
report["sha256"] = hashes
output = Path(__file__).with_name("source_verification.json")
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
for label in RUNS:
    print(label, json.dumps({k: v for k, v in report[label].items() if k != "aggregates"}))
print("Verified original V2 aggregates for all facets and both splits; wrote", output)
